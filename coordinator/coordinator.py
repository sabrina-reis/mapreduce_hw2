import rpyc
import time
import operator
import glob
import zipfile
import urllib.request
from pathlib import Path
from rpyc.utils.helpers import async_
import os
import collections

def split_text(globs, max_chars=2**23):
    """
    Splits input text files into chunks. Chunks are roughly even depending
    on word length to ensure work doesn't exceed time limit. 
    """
    batched_text = []
    buffer = ""
    punct = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    read_size = max_chars * 2
    
    for file in globs:
        with open(file, "r", encoding="utf-8") as f:
            while True:
                block = f.read(read_size)
                # break when no more text to read
                if not block:
                    break
                buffer += block

                # While buffer exceeds limit, extract a chunk
                while len(buffer) >= max_chars:
                    chunk = max_chars
                    # backtrack to nearest space or punctuation to avoid
                    # splitting words
                    while chunk > 0 and buffer[chunk - 1] not in punct and buffer[chunk - 1] != " ":
                        chunk -= 1
                    
                    # if no space found, word is too long--cut it short
                    if chunk == 0:
                        chunk = max_chars  

                    batched_text.append(buffer[:chunk].strip())
                    buffer = buffer[chunk:]

        # handle leftover text after finishing a file
        if buffer:
            batched_text.append(buffer.strip())
            buffer = ""
    return batched_text

def dispatcher(batched_text, pool, func="", poll_interval=1, timeout=20):
    """ 
    Async task dispatcher for mapingp batches to workers.
    Check every poll_interval if the workers are complete and start a new batch.
    Retry on a new worker from the pool if the worker takes too long. 
    RPC call must return list.
    """
    
    pending = list(batched_text)
    num_workers = len(pool)
    results = []
    timeouts = 0

    # Keep track of active workers and reassign tasks from stalled workers
    running = [None] * num_workers

    # confirm list of tasks is non-empty and >= 1 worker running
    while pending or any(running):

        if timeouts >= num_workers - 1 and not any(running):
            # If we reach this point it is because we timed out 
            # too many times.
            break
        
        # if we have failed too many times, the task is 
        # too large and has to be done on the coordinator.
        # However, we still need to loop until nothing else 
        # is running.
        if timeouts < num_workers - 1:
            # Assign batches to any idle workers
            for i, worker in enumerate(pool):

                # if no workers running and there are still tasks:
                if running[i] is None and pending:
                    print("Batching new work")
                    task = pending.pop(0) # pop off task from tasklist
                    # do non-blocking RPyC call to start worker
                    async_call = async_(getattr(worker.root, func))
                    # track running worker
                    running[i] = (task, async_call(task))

        # check if any workers have finished
        for i, entry in enumerate(running):
            if entry is not None:
                task, future = entry
                
                # add results from finished workers to array 
                if future.ready:
                        
                    running[i] = None

                    if len(future.value) > 0:
                        print("Copying completed work")
                        results.append(future.value)
                    
                    else:

                        found = False
                        timeouts += 1
                        
                        if timeouts < num_workers - 1:
                            # reassign task to another available worker
                            for j, worker in enumerate(pool):
                                # try to find other available worker
                                if i != j and running[j] is None:
                                    print(f"Task timed out, reassigning work from {i+1} to {j+1}")
                                    async_call = async_(getattr(worker.root, func))
                                    running[j] = (task, async_call(task))
                                    found = True
                                    break
                        
                        # if no other worker found, put task back in pool
                        # not perfect guarantee of retrying with different worker,
                        # but maybe current worker just needs a quick break :)
                        if not found:
                            print("No open workers, pending task.")
                            pending.append(task)    

        # don't check running tasks too frequently
        time.sleep(poll_interval)

    success = timeouts < num_workers - 1
    return results, pending, success

def mapreduce_wordcount(text, num_workers : int):
    """
    Driver function for mapreduce coordinator.
    """
    # Split text into roughly evenly sized batches
    print(f'Batching text into chunks')
    batched_text = split_text(text)
    
    # Connect to pool of workers using port specified in hw
    print(f"Connecting to pool of {num_workers} workers.")
    pool = []
    for i in range(num_workers):
        conn = rpyc.connect(f"mapreduce-worker-{i+1}", 18861)
        pool.append(conn)

    # Send batches to workers, recieve batched maps
    
    print(f"Mapping {len(batched_text)} chunks to {num_workers} workers.")
    batched_maps, _, _ = dispatcher(batched_text, pool, "map")

    reducing = True
    reduced_maps = []

    # Recursive reductions, e.g. reduce 8 lists of batched maps to 4, to 2, to 1.
    while reducing:
        print(f"Reducing {len(batched_maps)} maps using {num_workers} workers.")
        # list (in)comprehension: slice batched_maps into a list of two sublists for reduction
        # want each item to be a pair of sublists that can be reduced by worker
        chunked_maps = [
            batched_maps[i:min(i + 2, len(batched_maps) - 1)]
            for i in range(0, len(batched_maps), 2)
        ]
        # send list of paired lists to workers. NOTE: My Mac isn't 
        # fast enough to do the reductions on multiple threads, so 
        # no matter what we just use one.
        reduced_maps, pending, success = dispatcher(chunked_maps, pool, "reduce")
        
        if not success:
            print("Reduce workers taking too long. Reducing on coordinator.")
            batched_maps = reduced_maps

            # move all of the pending work
            for batch in pending:
                for b in batch:
                    batched_maps.append(b)
            
            # exit the loop
            break

        # since dispatcher returns a list of lists, final reduce should be 
        # list of one sublist of all reduced sums
        elif len(reduced_maps) < 4:
            reducing = False
        
        # update batched_maps to reduced values, e.g. 8 lists becomes 4
        else:
            batched_maps = reduced_maps
    
    print("Performing final reduction")
    final_reduce = collections.defaultdict(int)
    for partition in batched_maps:
        for key, value in partition:
            final_reduce[key] += value
    final_reduce = sorted(list(final_reduce.items()), key=operator.itemgetter(1), reverse=True)

    # Shutdown all workers 
    for c in pool:
        try:
            resp = c.root.shutdown()
            print(resp)
        except:
            print("Worker successfully terminated.")
    # return total_counts
    return final_reduce

# The Coordinator can exit when all Map and Reduce Tasks have finished.
# Number of Map and Reduce Workers should be configurable via command-line
# arguments or environment variables
def download(url="https://mattmahoney.net/dc/enwik9.zip"):
    """Downloads and unzips a wikipedia dataset in txt/."""
    zip_path = 'text.zip'
    work_dir = '/app/txt'
    Path("/app/txt").mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(work_dir)


if __name__ == "__main__":

    # specify URL to download dataset via OS environment variable
    url = os.getenv("STR_URL", "https://mattmahoney.net/dc/enwik8.zip")

    # specify num workers via environment variable
    num_workers = int(os.getenv("NUM_WORKERS", "4"))
    
    # Download and unzip dataset    
    download(url)
    
    start_time = time.time()
    
    input_files = glob.glob('/app/txt/*')
    print("Datasets to process:\n")
    for f in input_files:
        print(f)
    
    # call coordinator and workers to do mapreduce
    word_counts = mapreduce_wordcount(input_files, num_workers)
    
    # reporting word counts
    print('\nTOP 20 WORDS BY FREQUENCY\n')
    top20 = word_counts[0:20]
    longest = max(len(word) for word, _ in top20)
    
    i = 1
    for word, count in top20:
        print('%s.\t%-*s: %5s' % (i, longest+1, word, count))
        i = i + 1
    
    # reporting time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed Time: {} seconds".format(elapsed_time))

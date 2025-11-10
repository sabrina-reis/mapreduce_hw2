import rpyc
import string
import threading
import os
import time
from random import randint
import collections

class MapReduceService(rpyc.Service):
    def partition(self, mapped_values):
        """Internal function for intermediate partition/reduce within map function."""
        partitioned_data = collections.defaultdict(int)
        for key, value in mapped_values:
            partitioned_data[key] += value
        return partitioned_data


    def exposed_map(self, text_chunk):
        """
        Map step: tokenize and count words in text chunk.
        Taken from hw1.
        """

        # make translation dict to remove spaces
        TR = "".maketrans(string.punctuation, ' ' * len(string.punctuation))
        STOP_WORDS = set([
        'a', 'an', 'and', 'are', 'as', 'be', 'by', 'for', 'if', 'in',
        'is', 'it', 'of', 'or', 'py', 'rst', 'that', 'the', 'to', 'with',
        ])

        worker_id = randint(1, 1_000_000)
        mapped_values = []

        begin = time.time()

        # separate and count words
        print(f"{worker_id}: Starting map batch")
        lines = text_chunk.split('\n')
        for line in lines:

            if time.time() - begin > 20:
                print("Time limit exceeded")
                return []

            if line.lstrip().startswith('..'): # Skip rst comment lines
                continue
            
            line = line.translate(TR)     

            for word in line.split():
                word = word.lower()
                
                if word.isalpha() and word not in STOP_WORDS:
                    mapped_values.append((word, 1))
                    
        print(f"{worker_id}: Completed map batch")

        # According to hw, we do intermediate reduce here to get word counts
        print(f"{worker_id}: Starting reduce batch")
        partitioned_data = self.partition(mapped_values)
        if time.time() - begin > 20:
                print("Time limit exceeded")
                return []
        print(f"{worker_id}: Completed reduce batch")
    
        # return list of tuples of words and WCs for each worker
        return list(partitioned_data.items())
    
    def exposed_reduce(self, grouped_items):
        """Reduce step: sum counts for a subset of words."""
        worker_id = randint(1, int(1_000_000))
        print(f"{worker_id}: Starting reduce batch")
        partitioned_data = collections.defaultdict(int)
        begin = time.time()
        for partition in grouped_items:
            for key, value in partition:
                partitioned_data[key] += value
                if time.time() - begin > 20:
                    print("Time limit exceeded")
                    return []
        print(f"{worker_id}: Completed reduce batch")
        return list(partitioned_data.items())

    def exposed_shutdown(self):
        """Shut down this worker via RPC. Necessary due to cleanup issues."""
        # sleep to provide time to return and communicate exit to
        # coordinator. 
        threading.Thread(
            target=lambda: (time.sleep(0.1), os._exit(0)),
            daemon=True
        ).start()
        return None



if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(MapReduceService, port=18861)
    t.start()
"""
Distributed backend using multiprocessing pool.
"""
import multiprocessing
from .template import MetaBackend


class MultiprocessBackend(MetaBackend):
    def __init__(self, max_process=16):
        self.max_process = max_process
        if self.max_process > 1:
            self.pool = multiprocessing.Pool(processes=self.max_process)

    def __delete__(self, instance):
        self.pool.close()
        super().__delete__(instance)

    def distribute(self, function, params):
        if self.max_process == 1:
            results = [function(*param) for param in params]
        else:
            results = self.pool.starmap(function, params)
        return results

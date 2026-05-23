"""
Single thread backend.
"""
from .template import MetaBackend


class SingleProcessBackend(MetaBackend):
    def distribute(self, function, params):
        return [function(*param) for param in params]

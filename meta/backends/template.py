"""
Backend template for executing a function on each set of parameters.
"""


class MetaBackend:
    def __init__(self):
        pass

    def distribute(self, function, params):
        raise NotImplementedError(f"distribute not implemented for {type(self)}!")

"""
Tool to manage logging for series of experiments.
"""
from datetime import datetime, timedelta
import os
from string import ascii_lowercase
import time

from .log import log


class MultiLogger:
    def __init__(self, folder=None):
        self.folder = folder or "log"
        try:
            os.makedirs(self.folder)
            print(f"Created directory {self.folder}!")
        except FileExistsError:
            pass

        date = datetime.now()
        while any(
            date.strftime("%Y-%m-%d-%H-%M") in filename
            for filename in os.listdir(self.folder)
        ):
            date += timedelta(minutes=1)

        self.prefix = os.path.join(self.folder, f"{date.strftime('%Y-%m-%d-%H-%M')}")
        self.filename_generator = self.filename_gen()
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *_):
        print(f"Process took {time.time() - self.start_time} seconds.")

    def filename_gen(self):
        for letter1 in ["_"] + list(ascii_lowercase):
            for letter2 in ["_"] + list(ascii_lowercase):
                for letter3 in ["_"] + list(ascii_lowercase):
                    for letter4 in ["_"] + list(ascii_lowercase):
                        for letter5 in ascii_lowercase:
                            yield f"{self.prefix}-{letter1}{letter2}{letter3}{letter4}{letter5}.json"

    def summarize(self, results=None, info=None, filename_extension="SUMMARY"):
        filename = f"{self.prefix}~{filename_extension}.json"
        log(None, None, results, info, filename=filename)

    def log(self, network, game, results=None, info=None, log_fn=log):
        return log_fn(network, game, results, info, filename=next(self.filename_generator))

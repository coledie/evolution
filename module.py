"""
The base Module definition. Provides structure and functionality
that every class should abide by.
"""
from copy import deepcopy
import pickle
import numpy as np


class Key:
    def __init__(self, name, description, type=any, default="veryspecificstring"):
        self.name = name
        self.type = type
        if self.type == float:
            self.type = (float, int)
        self.description = description
        if default != "veryspecificstring":
            self.default = default

    def __str__(self):
        t = self.type if self.type != any else "any"
        default_str = f", default={self.default}" if hasattr(self, "default") else ""
        return f'"{self.name}": "[{t}{default_str}] {self.description}"'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, Key):
            return self.name == other.name
        return self.name == other

    def __hash__(self):
        return hash(self.name)


class Module:
    NECESSARY_KEYS = []

    def __init__(self, **kwargs):
        self._add_values(kwargs)
        self.training = True

    @classmethod
    def extend_keys(cls, new_keys, base="NECESSARY_KEYS"):
        keys = getattr(cls, base)
        keys = deepcopy(keys)
        if isinstance(keys, dict):
            if isinstance(new_keys, list):
                new_keys = {key.name: key.description for key in new_keys}
            keys.update(new_keys)
        elif isinstance(keys, list):
            if isinstance(new_keys, dict):
                new_keys = [Key(key, description) for key, description in new_keys.items()]
            keys.extend(new_keys)
        return keys

    @classmethod
    def list_keys(cls):
        print("{")
        for key in cls.NECESSARY_KEYS:
            if isinstance(key, Key):
                print(f"\t{str(key)},")
            else:
                desc = cls.NECESSARY_KEYS[key]
                print(f"\t{key}: {desc},")
        print("}")

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def copy(self):
        return deepcopy(self)

    def _check_config(self, kwargs, base="NECESSARY_KEYS"):
        missing = []
        for key in getattr(self, base):
            if isinstance(key, Key):
                name = key.name
                if name not in kwargs and not hasattr(key, "default"):
                    missing.append(name)
            elif isinstance(key, str):
                if key not in kwargs:
                    missing.append(key)
        if missing:
            raise KeyError(f"Missing values for keys {missing}, all of which do not have defaults!")

    def _add_values(self, kwargs, base="NECESSARY_KEYS", dest=None, prefix="_"):
        dest = dest or self
        if isinstance(dest, str):
            dest = getattr(self, dest)
        self._check_config(kwargs, base)

        for key in getattr(self, base):
            if isinstance(key, Key):
                name = key.name
                if name in kwargs:
                    value = kwargs[name]
                    if key.type == np.array:
                        key.type = np.ndarray
                    if key.type != any and not isinstance(value, key.type):
                        if key.type is np.ndarray and isinstance(value, (list, tuple)):
                            value = np.array(value)
                        else:
                            raise KeyError(f"Key {name} is incorrect type, got {type(value)} and expected {key.type}!")
                else:
                    if not hasattr(key, "default"):
                        raise KeyError(f"No value given for key, '{name}'!")
                    value = key.default
            elif isinstance(key, str):
                name = key
                value = kwargs[name]

            if isinstance(dest, dict):
                dest[f"{prefix}{name}"] = value
            else:
                setattr(dest, f"{prefix}{name}", value)


def save(module, filename, pickle_module=pickle, pickle_protocol=2):
    with open(filename, "wb") as file:
        pickle_module.dump(module, file, protocol=pickle_protocol)


def load(filename, pickle_module=pickle):
    with open(filename, "rb") as file:
        module = pickle_module.load(file)
    if not isinstance(module, Module):
        raise ValueError("Cannot load this type of file.")
    return module

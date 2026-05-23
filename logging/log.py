"""
Core experiment logging functionality.
"""
from datetime import datetime
import json
import os

import numpy as np

from .sanitize import sanitize_dictionary


def log(
    network=None,
    game=None,
    results=None,
    info=None,
    folder="",
    filename=None,
    **kwargs,
):
    if not folder and not filename:
        folder = os.path.join(os.path.abspath(__file__).split("evolution")[0], "log")
    if folder:
        try:
            os.makedirs(folder)
            print(f"Created directory {folder}!")
        except FileExistsError:
            pass
    if filename is None:
        filename = f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}.json"
    filename = os.path.join(folder, filename)

    data = {}

    if game is not None:
        game_params = {"name": str(type(game))}
        game_params.update(game.params)
        data.update({"game": sanitize_dictionary(game_params)})

    if network is not None:
        snn_params = {}
        for key, value in network.parts.items():
            if hasattr(value, "__name__"):
                snn_params.update({key: value.__name__})
            else:
                snn_params.update({key: str(type(value).__name__)})
        snn_params.update(network.params)
        data.update({"snn": sanitize_dictionary(snn_params)})

    data = sanitize_dictionary(data)

    if results is not None:
        data.update({"results": sanitize_dictionary(results)})
    if info is not None:
        data.update({"info": sanitize_dictionary(info)})

    order_rankings = {
        int: 0, float: 0, str: 20, tuple: 40,
        "default": 50, list: 60, set: 70,
        np.ndarray: 99, np.ma.core.MaskedArray: 99,
    }

    for data_key, values in data.items():
        if not isinstance(values, dict):
            continue
        evaluations = {}
        for key, value in values.items():
            vt = type(value)
            evaluations[key] = order_rankings.get(vt, 50)
        ordering = sorted(evaluations, key=evaluations.get)
        data[data_key] = {key: values[key] for key in ordering}

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    return filename

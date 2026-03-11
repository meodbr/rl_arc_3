from typing import Any, Iterable
import os
import json
import logging.config
from queue import Empty as EmptyQueueException
from queue import Full as FullQueueException

import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event

from rl_arc_3.settings import settings

def linear_interp(tau, a, b):
    return (tau) * b + (1 - tau) * a

def get_model_device(model: nn.Module) -> str:
    return next(model.parameters()).device

def push_with_stop(
    queue: mp.Queue, item: Any, stop_events: Iterable[Event] | Event, timeout: float = 0.1
) -> bool:
    """
    Push an item to a multiprocessing queue with a timeout, checking for a stop event.
    """
    events = [stop_events] if isinstance(stop_events, Event) else stop_events
    pushed = False
    while not pushed:
        try:
            queue.put(item, timeout=timeout)
            pushed = True
        except FullQueueException:
            if any(stop_event.is_set() for stop_event in events):
                break
    return pushed

def get_with_stop(
    queue: mp.Queue, stop_events: Iterable[Event] | Event, timeout: float = 0.1
) -> Any:
    """
    Get an item from a multiprocessing queue with a timeout, checking for a stop event.
    """
    events = [stop_events] if isinstance(stop_events, Event) else stop_events
    while True:
        try:
            item = queue.get(timeout=timeout)
            return item
        except EmptyQueueException:
            if any(stop_event.is_set() for stop_event in events):
                return None

def setup_logging(config_path: str = settings.LOGGING_CONFIG):
    os.makedirs("logs", exist_ok=True)

    with open(config_path) as f:
        config = json.load(f)

    process_name = mp.current_process().name
    logfile = f"logs/{process_name}.log"

    if "handlers" in config and "file_per_process" in config["handlers"]:
        config["handlers"]["file_per_process"]["filename"] = logfile

    logging.config.dictConfig(config)

def unwrap_if_single(x: Iterable[Any]) -> Any:
    if len(x) == 1:
        return x[0]
    return x
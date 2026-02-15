from typing import Any
from queue import Empty as EmptyQueueException
from queue import Full as FullQueueException

import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event

def linear_interp(tau, a, b):
    return (tau) * b + (1 - tau) * a

def get_model_device(model: nn.Module) -> str:
    return next(model.parameters()).device

def push_with_stop(
    queue: mp.Queue, item: Any, stop_event: Event, timeout: float = 0.1
) -> bool:
    """
    Push an item to a multiprocessing queue with a timeout, checking for a stop event.
    """
    pushed = False
    while not pushed:
        try:
            queue.put(item, timeout=timeout)
            pushed = True
        except FullQueueException:
            if stop_event.is_set():
                break
    return pushed

def get_with_stop(
    queue: mp.Queue, stop_event: Event, timeout: float = 0.1
) -> Any:
    """
    Get an item from a multiprocessing queue with a timeout, checking for a stop event.
    """
    while True:
        try:
            item = queue.get(timeout=timeout)
            return item
        except EmptyQueueException:
            if stop_event.is_set():
                return None
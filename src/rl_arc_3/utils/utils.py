from typing import Any

import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized

def linear_interp(tau, a, b):
    return (tau) * b + (1 - tau) * a

def get_model_device(model: nn.Module) -> str:
    return next(model.parameters()).device

def push_with_stop(
    queue: mp.Queue, item: Any, stop_event: Synchronized[Any], timeout: float = 0.1
) -> bool:
    """
    Push an item to a multiprocessing queue with a timeout, checking for a stop event.
    """
    pushed = False
    while not pushed:
        try:
            queue.put(item, timeout=timeout)
            pushed = True
        except mp.queues.Full:
            if stop_event.is_set():
                break
    return pushed

def get_with_stop(
    queue: mp.Queue, stop_event: Synchronized[Any], timeout: float = 0.1
) -> Any:
    """
    Get an item from a multiprocessing queue with a timeout, checking for a stop event.
    """
    while True:
        try:
            item = queue.get(timeout=timeout)
            return item
        except mp.queues.Empty:
            if stop_event.is_set():
                return None
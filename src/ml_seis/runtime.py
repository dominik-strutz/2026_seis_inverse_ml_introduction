"""Small runtime helpers for the notebooks."""

from contextlib import contextmanager

import torch


@contextmanager
def torch_threads(n: int):
    """Run a block, or a whole function used as a decorator, with ``n`` torch
    threads. The tiny networks of the course train fastest on one thread: splitting a
    small matrix product over threads costs more than it saves."""
    before = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(before)

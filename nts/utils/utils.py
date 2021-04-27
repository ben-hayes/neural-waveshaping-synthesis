import os
from typing import Callable, Sequence


def apply(fn: Callable[[any], any], x: Sequence[any]):
    if type(x) not in (tuple, list):
        raise TypeError("x must be a tuple or list.")
    return type(x)([fn(element) for element in x])


def apply_unpack(fn: Callable[[any], any], x: Sequence[Sequence[any]]):
    if type(x) not in (tuple, list):
        raise TypeError("x must be a tuple or list.")
    return type(x)([fn(*element) for element in x])


def unzip(x: Sequence[any]):
    return list(zip(*x))


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

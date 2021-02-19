from typing import Sequence

def unzip(x: Sequence[any]):
    return list(zip(*x))
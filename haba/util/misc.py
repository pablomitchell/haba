"""
Miscellaneous utilities
"""

import functools
import time

def timer(func):
    """
    Print the runtime of the decorated function
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        number = 10
        start_time = time.perf_counter()

        for _ in range(number):
            value = func(*args, **kwargs)

        end_time = time.perf_counter()
        ave_time = (end_time - start_time) / number
        print(
            f'Timer: {func.__name__!r} averaged {ave_time:.4f} '
            f'seconds over {number} iterations.'
        )
        return value

    return wrapper_timer

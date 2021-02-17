"""
Simple multi-processing tools
"""

from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


def amap(func, args, n_workers=None, **kwargs):
    """
    Asynchronous map

    Computes func using each element of an iterable as argument.
    Each element maps one-to-one to a process submitted via the
    concurrent.futures module.  Additional arguments required by
    func may be supplied via kwargs.  Progress of amap is shown
    via tqdm progress bar.

    Parameters
    ----------
    func : callable
        method to execute asynchronously
    args : iterable
        each element is supplied to func as a separate argument
    n_works : int, default None
        number of workers (processes) to use in parallel which,
        if not supplied, defaults to the system number of cores
    kwargs : dict
        additional arguments passed to func

    Returns
    -------
    results : dict
        each key corresponds 1-to-1 with elements in args and the
        values are the results of func being called on each key

    Example
    -------
    >>> def really_slow_computation(x, y=False):  return some_func(x, y)
    >>> long_list_of_inputs = generate_a_list_somehow()
    >>> results = amap(really_slow_computation, long_list_of_inputs, y=True)

    """
    n_jobs = len(iterable)
    futures = {}

    with tqdm(total=n_jobs, ncols=80) as progress:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for ii in iterable:
                futures[ii] = pool.submit(single_task, ii, **kwargs)
                futures[ii].add_done_callback(lambda p: progress.update())

    results = {k: f.result() for k, f in futures.items()}

    return results

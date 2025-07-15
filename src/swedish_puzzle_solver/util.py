
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Final
import logging
import traceback

LOG:Final[logging.Logger] = logging.getLogger(__name__)

def run_functions_in_parallel(
    functions: List[Callable[[Any, Any], List[str]]],
    arg1: Any,
    arg2: Any,
    max_workers: int = 3,
) -> List[str]:
    """
    Executes a list of functions in parallel, each accepting two arguments and returning a list of strings.

    :param functions: List of functions to execute.
    :param arg1: First argument passed to each function.
    :param arg2: Second argument passed to each function.
    :param max_workers: Maximum number of threads to run concurrently.
    :return: Combined list of strings returned from all functions.
    """
    results: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_function = {
            executor.submit(func, arg1, arg2): func for func in functions
        }

        for future in as_completed(future_to_function):
            func = future_to_function[future]
            try:
                result = future.result()
                results.extend(result)
            except Exception as e:
                traceback.print_exc()
                LOG.exception(f"Function '{func.__name__}' raised an exception: {e}")

    return results
import time


class Logger:
    """
    Lightweight structured logger for the KFC pipeline.

    This logger provides a minimal, dependency-free logging utility
    designed specifically for debugging and profiling multi-stage
    machine learning pipelines (K-step → F-step → C-step).

    It supports hierarchical verbosity levels similar to logging
    libraries, but remains simple and NumPy-friendly.

    Logging levels
    --------------
    verbose = 0
        Silent mode (no output)

    verbose = 1
        Basic information messages (high-level pipeline events)

    verbose = 2
        Debug messages (shapes, stage transitions, model info)

    verbose = 3
        Trace-level output (iteration-level and fine-grained updates)

    Parameters
    ----------
    verbose : int, default=0
        Controls the logging verbosity level.

    Attributes
    ----------
    verbose : int
        Current verbosity level.

    t0 : float
        Timestamp marking logger initialization time (used for elapsed time tracking).

    Notes
    -----
    - This logger is intentionally minimal and does not depend on
      Python's standard `logging` module.
    - All timestamps are relative to the logger initialization time.
    - Designed for ML experimentation, not production logging.

    Examples
    --------
    >>> logger = KFCLogger(verbose=2)
    >>> logger.info("Training started")
    >>> logger.debug("Shape = (100, 10)")
    >>> logger.trace("Iteration 1 complete")
    """

    def __init__(self, verbose: int = 0):
        """
        Initialize the KFC logger.

        Parameters
        ----------
        verbose : int, default=0
            Logging verbosity level (0–3).
        """
        self.verbose = verbose
        self.t0 = time.time()

    def log(self, level: int, msg: str):
        """
        Internal logging dispatcher.

        Parameters
        ----------
        level : int
            Required verbosity level to display the message.
        msg : str
            Message to display.

        Notes
        -----
        If `self.verbose >= level`, the message is printed with
        elapsed time since initialization.
        """
        if self.verbose >= level:
            elapsed = time.time() - self.t0
            print(f"[KFC][{elapsed:8.3f}s] {msg}")

    def info(self, msg: str):
        """
        Log high-level pipeline information.

        Parameters
        ----------
        msg : str
            Informational message.
        """
        self.log(1, msg)

    def debug(self, msg: str):
        """
        Log debugging information.

        Parameters
        ----------
        msg : str
            Debug message (e.g., shapes, shapes, model states).
        """
        self.log(2, msg)

    def trace(self, msg: str):
        """
        Log fine-grained execution details.

        Parameters
        ----------
        msg : str
            Trace-level message (e.g., per-iteration updates).
        """
        self.log(3, msg)

def timed(logger, name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.debug(f"[{name}] start")
            result = func(*args, **kwargs)
            logger.debug(f"[{name}] done in {time.time()-start:.4f}s")
            return result
        return wrapper
    return decorator
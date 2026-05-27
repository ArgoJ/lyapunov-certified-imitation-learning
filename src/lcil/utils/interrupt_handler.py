import time
import sys
import logging

class GracefulInterruptHandler:
    """Context manager to handle KeyboardInterrupt gracefully, allowing for a double Ctrl+C to exit immediately."""

    _last_interrupt_time: float = 0.0

    def __init__(self, *, delta: float = 3.0, logger: logging.Logger | None = None):
        """Contruct a GracefulInterruptHandler.

        Parameters
        ----------
        delta : float, optional
            Time window in seconds to detect a double Ctrl+C, by default 3.0
        logger : logging.Logger | None, optional
            Logger instance to use for logging messages, by default None
        """
        self.aborted = False
        self.logger = logger or logging.getLogger(__name__)
        self.delta = delta

    def __enter__(self):
        self.aborted = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):        
        if exc_type is KeyboardInterrupt:
            current_time = time.time()
            current_delta = current_time - GracefulInterruptHandler._last_interrupt_time

            # double press Ctrl+C 
            if self.delta <= 0 or current_delta < self.delta:
                self.logger.error("Second interrupt detected within %.1f seconds. Exiting immediately.", current_delta)
                sys.exit(1)
            
            else:
                GracefulInterruptHandler._last_interrupt_time = current_time
                self.logger.warning("Skipping current operation. Press Ctrl+C again within %.1f seconds to exit the script immediately.", self.delta)
                self.aborted = True
                return True 
        return False
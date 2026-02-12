import logging
import sys

from tqdm import tqdm
from contextlib import contextmanager

DEFAULT_MODULE_NAME = "mpc_datagen"
DEFAULT_SHORT_NAME = "mdg"
DEFAULT_LOGGER_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s'


class ShortNameFormatter(logging.Formatter):
    """
    Formatter that replaces the long package name with a short one 
    in the log output without breaking the logger hierarchy.
    """
    def format(self, record):
        original_name = record.name
        if record.name.startswith(DEFAULT_MODULE_NAME):
            record.name = record.name.replace(DEFAULT_MODULE_NAME, DEFAULT_SHORT_NAME)
            
        result = super().format(record)
        record.name = original_name
        return result


class TqdmLoggingHandler(logging.Handler):
    """
    A logging handler that writes logs via tqdm.write, so they don't corrupt the progress bar.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout) 
        except Exception:
            self.handleError(record)


class PackageLogger(logging.Logger):
    """
    Custom Logger class that includes a context manager for tqdm-safe logging.
    """
    
    @contextmanager
    def tqdm(self, **tqdm_kwargs):
        """
        Context manager to safely wrap a loop with a progress bar while logging.
        
        Usage
        -----
        ```python
        logger = logging.getLogger("my_module")
        with logger.tqdm(total=100) as pbar:
            for i in range(100):
                ...
        ```
        """
        target_logger = self
        if not self.handlers and self.propagate and self.parent:
            target_logger = logging.getLogger(DEFAULT_MODULE_NAME)

        tqdm_handler, restored_handlers = self._swap_to_tqdm_handler(target_logger)
        pbar = tqdm(**tqdm_kwargs)
        
        try:
            yield pbar
        finally:
            pbar.close()
            if tqdm_handler:
                self._restore_handlers(target_logger, tqdm_handler, restored_handlers)

    def _swap_to_tqdm_handler(self, logger_instance: logging.Logger):
        """Internal helper to swap StreamHandlers with TqdmLoggingHandler."""
        removed_handlers = []
        
        # Delete StreamHandler 
        for h in list(logger_instance.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, TqdmLoggingHandler):
                logger_instance.removeHandler(h)
                removed_handlers.append(h)
        
        # Add Tqdm Handler
        handler = TqdmLoggingHandler()
        formatter = ShortNameFormatter(DEFAULT_LOGGER_FORMAT)
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)
        
        return handler, removed_handlers

    def _restore_handlers(self, logger_instance: logging.Logger, handler_to_remove, handlers_to_restore):
        """Internal helper to restore original handlers."""
        logger_instance.removeHandler(handler_to_remove)
        for h in handlers_to_restore:
            logger_instance.addHandler(h)

    @staticmethod
    def setup(package_name: str = DEFAULT_MODULE_NAME, level: int = logging.INFO):
        """
        Registers this class as the default Logger and sets up the root package logger.
        MUST be called before any logging.getLogger() calls in the main script.
        """
        # Register PackageLogger as the default Logger class
        logging.setLoggerClass(PackageLogger)
        
        # Configure root package logger
        logger = logging.getLogger(package_name)
        logger.setLevel(level)
        logger.propagate = False # Prevent duplicate logs to the root system logger
        
        # Remove old handlers (reset)
        if logger.handlers:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)

        # Standard Handler (Console)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = ShortNameFormatter(DEFAULT_LOGGER_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
import logging
import sys
from tqdm import tqdm

DEFAULT_MODULE_NAME = "lyapunov_certified_imitation_learning"
DEFAULT_SHORT_NAME = "lcil" 
DEFAULT_LOGGER_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s'


class ShortNameFormatter(logging.Formatter):
    """
    Formatter that replaces the long package name with 'lcil' in the log output
    without breaking the logger hierarchy.
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
            tqdm.write(msg, file=sys.__stdout__)
        except Exception:
            self.handleError(record)


class PackageLogger:
    """
    Configuration utility for the package logger.
    """
    @staticmethod
    def setup(package_name: str = DEFAULT_MODULE_NAME, level: int = logging.INFO) -> logging.Logger:
        """
        Sets up the root logger for the package with a default StreamHandler.
        Resets existing handlers to ensure configuration updates are applied.
        """
        logger = logging.getLogger(package_name)
        logger.setLevel(level)
        logger.propagate = False
        
        # Remove existing handlers to ensure fresh configuration
        if logger.handlers:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)

        # Add the default handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = ShortNameFormatter(DEFAULT_LOGGER_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
            
        return logger

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Retrieves a logger with the specified name within the package.
        """
        return logging.getLogger(name)

    @staticmethod
    def add_tqdm_handler(package_name: str = DEFAULT_MODULE_NAME) -> logging.Handler:
        """
        Adds a TqdmLoggingHandler to the package logger and removes other StreamHandlers 
        to prevent duplicate output. Returns the added handler.
        """
        logger = logging.getLogger(package_name)
        
        # Remove existing StreamHandlers (assuming they print to stdout/stderr)
        removed_handlers = []
        for h in list(logger.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, TqdmLoggingHandler):
                logger.removeHandler(h)
                removed_handlers.append(h)
        
        handler = TqdmLoggingHandler()
        formatter = ShortNameFormatter(DEFAULT_LOGGER_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return handler, removed_handlers

    @staticmethod
    def restore_handlers(package_name: str, handler_to_remove: logging.Handler, handlers_to_restore: list):
        """
        Restores the previous handlers and removes the TqdmLoggingHandler.
        """
        logger = logging.getLogger(package_name)
        logger.removeHandler(handler_to_remove)
        for h in handlers_to_restore:
            logger.addHandler(h)

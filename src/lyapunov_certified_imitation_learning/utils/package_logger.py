import logging
import sys
from tqdm import tqdm

_DEFAULT_LOGGER_NAME = "lcil"
_DEFAULT_LOGGER_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s'

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
    def setup(package_name: str = _DEFAULT_LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
        """
        Sets up the root logger for the package with a default StreamHandler.
        Resets existing handlers to ensure configuration updates are applied.
        """
        logger = logging.getLogger(package_name)
        logger.setLevel(level)
        
        # Remove existing handlers to ensure fresh configuration
        if logger.handlers:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)

        # Add the default handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(_DEFAULT_LOGGER_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
            
        return logger

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Parameters
        ----------
        name : str
            The name of the module (e.g., __name__).
        
        Returns
        -------
        logging.Logger
            A logger instance.
        """
        if name.startswith("lyapunov_certified_imitation_learning"):
            name = name.replace("lyapunov_certified_imitation_learning", _DEFAULT_LOGGER_NAME, 1)

        return logging.getLogger(name)

    @staticmethod
    def add_tqdm_handler(package_name: str = _DEFAULT_LOGGER_NAME) -> logging.Handler:
        """
        Adds a TqdmLoggingHandler to the package logger and removes other StreamHandlers 
        to prevent duplicate output. Returns the added handler.
        """
        logger = logging.getLogger(package_name)
        
        # Remove existing StreamHandlers (assuming they print to stdout/stderr)
        # We keep FileHandlers etc.
        removed_handlers = []
        for h in list(logger.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, TqdmLoggingHandler):
                logger.removeHandler(h)
                removed_handlers.append(h)
        
        handler = TqdmLoggingHandler()
        formatter = logging.Formatter(_DEFAULT_LOGGER_FORMAT)
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

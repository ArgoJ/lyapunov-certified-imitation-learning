import logging
import sys

from tqdm import tqdm
from contextlib import contextmanager

DEFAULT_MODULE_NAME = "lyapunov_certified_imitation_learning"
DEFAULT_SHORT_NAME = "lcil"
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


class PackageLogger:
    """
    Package-local logging utilities.
    """

    @contextmanager
    def tqdm(logger: logging.Logger, **tqdm_kwargs):
        """Context manager to safely wrap a loop with tqdm-aware logging."""
        target_logger = logger
        if not logger.handlers and logger.propagate and logger.parent:
            target_logger = logging.getLogger(DEFAULT_MODULE_NAME)

        tqdm_handler, restored_handlers = PackageLogger._swap_to_tqdm_handler(target_logger)
        pbar = tqdm(**tqdm_kwargs)

        try:
            yield pbar
        finally:
            pbar.close()
            if tqdm_handler:
                PackageLogger._restore_handlers(target_logger, tqdm_handler, restored_handlers)

    @staticmethod
    def _swap_to_tqdm_handler(logger_instance: logging.Logger):
        """Internal helper to swap StreamHandlers with TqdmLoggingHandler."""
        removed_handlers = []

        for h in list(logger_instance.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, TqdmLoggingHandler):
                logger_instance.removeHandler(h)
                removed_handlers.append(h)

        handler = TqdmLoggingHandler()
        formatter = ShortNameFormatter(DEFAULT_LOGGER_FORMAT)
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)

        return handler, removed_handlers

    @staticmethod
    def _restore_handlers(logger_instance: logging.Logger, handler_to_remove, handlers_to_restore):
        """Internal helper to restore original handlers."""
        logger_instance.removeHandler(handler_to_remove)
        for h in handlers_to_restore:
            logger_instance.addHandler(h)

    @staticmethod
    def setup(package_name: str = DEFAULT_MODULE_NAME, level: int = logging.INFO):
        """
        Configure only the package root logger.
        """
        logger = logging.getLogger(package_name)
        logger.setLevel(level)
        logger.propagate = False

        if logger.handlers:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = ShortNameFormatter(DEFAULT_LOGGER_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger


class PackageBoundLogger(logging.LoggerAdapter):
    """LoggerAdapter with tqdm support for package-local logging."""

    def __init__(self, logger: logging.Logger):
        super().__init__(logger, extra={})

    @contextmanager
    def tqdm(self, **tqdm_kwargs):
        with PackageLogger.tqdm(self.logger, **tqdm_kwargs) as pbar:
            yield pbar


def get_package_logger(name: str) -> PackageBoundLogger:
    """Return a package-scoped logger wrapper with `.tqdm(...)` support."""
    return PackageBoundLogger(logging.getLogger(name))
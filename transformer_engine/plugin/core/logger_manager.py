# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import logging
import sys
import os
import threading

class Logger:
    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        formatter = logging.Formatter(
            "[%(asctime)s %(name)s %(filename)s:%(lineno)d %(levelname)s] %(message)s"
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)
        self._printed_once = set()

    def info(self, message):
        self.logger.info(message, stacklevel=2)

    def warning(self, message):
        self.logger.warning(message, stacklevel=2)

    def error(self, message):
        self.logger.error(message, stacklevel=2)

    def critical(self, message):
        self.logger.critical(message, stacklevel=2)

    def debug(self, message):
        self.logger.debug(message, stacklevel=2)

    def info_once(self, message):
        if message not in self._printed_once:
            self._printed_once.add(message)
            self.logger.info(message, stacklevel=2)

    def warning_once(self, message):
        if message not in self._printed_once:
            self._printed_once.add(message)
            self.logger.warning(message, stacklevel=2)

    def error_once(self, message):
        if message not in self._printed_once:
            self._printed_once.add(message)
            self.logger.error(message, stacklevel=2)

    def debug_once(self, message):
        if message not in self._printed_once:
            self._printed_once.add(message)
            self.logger.debug(message, stacklevel=2)

class LoggerManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        if hasattr(self, '_global_logger'):
            return

        self._global_logger = None
        self._global_printed_once = set()
        self._printed_once_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__()
        return cls._instance

    def get_logger(self):
        if self._global_logger is None:
            with self._lock:
                if self._global_logger is None:
                    level = os.getenv("TEFL_LOG_LEVEL", "INFO").upper()
                    self._global_logger = Logger("TE-FL", level)
        return self._global_logger

    def print_once(self, message):
        with self._printed_once_lock:
            if message not in self._global_printed_once:
                self._global_printed_once.add(message)
                print(message)

    def debug_print_once(self, func_name: str, backend_name: str = "Backend", *args, **kwargs):
        key = f"{backend_name}.{func_name}"

        with self._printed_once_lock:
            if key not in self._global_printed_once:
                self._global_printed_once.add(key)
                print(f"[{backend_name}] Calling {func_name}")
                if args:
                    print(f"  args: {[type(a).__name__ for a in args[:5]]}...")
                if kwargs:
                    print(f"  kwargs: {list(kwargs.keys())[:5]}...")
                print(f"[{backend_name}] {func_name} completed successfully")

    def reset(self):
        with self._lock:
            with self._printed_once_lock:
                self._global_logger = None
                self._global_printed_once.clear()

def get_logger():
    return LoggerManager.get_instance().get_logger()

def print_once(message):
    LoggerManager.get_instance().print_once(message)

def debug_print_once(func_name: str, backend_name: str = "Backend", *args, **kwargs):
    LoggerManager.get_instance().debug_print_once(func_name, backend_name, *args, **kwargs)
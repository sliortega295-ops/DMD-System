"""Keep ordinary CLI diagnostics separate from the final stdout receipt."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, redirect_stdout
import logging
import sys


def _restore_stream(handler, stream):
    # setStream flushes the temporary target. Do not let a failing flush mask
    # a plugin exception or prevent the remaining handlers from being restored.
    handler.acquire()
    try:
        handler.stream = stream
    finally:
        handler.release()


@contextmanager
def redirect_diagnostics():
    """Temporarily route Python stdout and its existing log handlers to stderr."""
    stdout, stderr = sys.stdout, sys.stderr
    loggers = [logging.getLogger(), *list(logging.Logger.manager.loggerDict.values())]
    with ExitStack() as stack:
        stack.enter_context(redirect_stdout(stderr))
        for logger in loggers:
            if not isinstance(logger, logging.Logger):  # Skip namespace placeholders.
                continue
            for handler in tuple(logger.handlers):
                if isinstance(handler, logging.StreamHandler) and handler.stream is stdout:
                    # Register rollback before setStream: its entry flush can
                    # fail after earlier handlers have already been retargeted.
                    stack.callback(_restore_stream, handler, stdout)
                    handler.setStream(stderr)
        yield

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines a logging class based on the built-in logging module.
The module is heavily based on the astropy logging system.
"""

from __future__ import print_function

import logging
from logging import FileHandler
from logging.handlers import TimedRotatingFileHandler

import os
import re
import shutil
import sys
import warnings

from textwrap import TextWrapper

from .colour_print import colour_print


# Initialize by calling initLog()
log = None

# Adds custom log level for important messages
IMPORTANT = 25
logging.addLevelName(IMPORTANT, 'IMPORTANT')

ansi_escape = re.compile(r'\x1b[^m]*m')


def important(self, message, *args, **kws):
    self._log(IMPORTANT, message, args, **kws)


logging.Logger.important = important


def get_log(logger_name, log_file_path=None, log_level='INFO',
            log_file_level='DEBUG', log_file_mode='a'):

    log_file_path = log_file_path or f'~/.{logger_name}/{logger_name}.log'
    logging.setLoggerClass(Logger)

    log = logging.getLogger(logger_name)
    log._set_defaults(
        logLevel=log_level,
        logFileLevel=log_file_level,
        logFilePath=log_file_path,
        logFileMode=log_file_mode)

    return log


class MyFormatter(logging.Formatter):

    warning_fmp = '%(asctime)s - %(levelname)s: %(message)s [%(origin)s]'
    info_fmt = '%(asctime)s - %(levelname)s: %(message)s'

    def __init__(self, fmt='%(levelno)s: %(msg)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.INFO:
            self._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.WARNING:
            self._fmt = MyFormatter.warning_fmp

        elif record.levelno == IMPORTANT:
            self._fmt = MyFormatter.info_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result


Logger = logging.getLoggerClass()
fmt = MyFormatter()


class Logger(Logger):
    """This class is used to set up the logging system.

    The main functionality added by this class over the built-in
    logging.Logger class is the ability to keep track of the origin of the
    messages, the ability to enable logging of warnings.warn calls and
    exceptions, and the addition of colorized output and context managers to
    easily capture messages to a file or list.

    """

    def saveLog(self, path):
        shutil.copyfile(self.logFilename, os.path.expanduser(path))

    def _showwarning(self, *args, **kwargs):

        warning = args[0]
        message = '{0}: {1}'.format(warning.__class__.__name__, args[0])
        mod_path = args[2]

        mod_name = None
        mod_path, ext = os.path.splitext(mod_path)
        for name, mod in sys.modules.items():
            path = os.path.splitext(getattr(mod, '__file__', ''))[0]
            if path == mod_path:
                mod_name = mod.__name__
                break

        if mod_name is not None:
            self.warning(message, extra={'origin': mod_name})
        else:
            self.warning(message)

    def _stream_formatter(self, record, wrap=False):
        """The formatter for standard output."""
        if record.levelno < logging.DEBUG:
            print(record.levelname, end='')
        elif(record.levelno < logging.INFO):
            colour_print(record.levelname, 'green', end='')
        elif(record.levelno < IMPORTANT):
            colour_print(record.levelname, 'magenta', end='')
        elif(record.levelno < logging.WARNING):
            colour_print(record.levelname, 'lightblue', end='')
        elif(record.levelno < logging.ERROR):
            colour_print(record.levelname, 'brown', end='')
        else:
            colour_print(record.levelname, 'red', end='')

        record.msg = ansi_escape.sub('', record.msg)

        record.message = '{0}'.format(record.msg)
        if record.levelno == logging.WARN:
            record.message = '{0}'.format(record.msg[record.msg.find(':') + 2:])

        # if not hasattr(record, 'origin') or record.origin == '':
        #     record.message = '{0}'.format(record.msg)
        # else:
        #     record.message = '{0} [{1:s}]'.format(record.msg, record.origin)

        if self.wrap is True and len(record.message) > 80:
            tw = TextWrapper()
            tw.width = 100
            tw.subsequent_indent = ' ' * (len(record.levelname) + 2)
            tw.break_on_hyphens = False
            msg = '\n'.join(tw.wrap(record.message))
            print(': ' + msg)
        else:
            print(': ' + record.message)

    def _set_defaults(self,
                      logLevel='WARNING',
                      logFileLevel='INFO',
                      logFilePath='~/.platedesign/platedesign.log',
                      logFileMode='w',
                      wrap=False):
        """Reset logger to its initial state."""

        # Remove all previous handlers
        for handler in self.handlers[:]:
            self.removeHandler(handler)

        # Set levels
        self.setLevel('DEBUG')

        # Set up the stdout handler
        sh = logging.StreamHandler()
        sh.emit = self._stream_formatter
        self.wrap = wrap
        self.addHandler(sh)
        sh.setLevel(logLevel)

        # Set up the main log file handler if requested (but this might fail if
        # configuration directory or log file is not writeable).

        logFilePath = os.path.expanduser(logFilePath)
        logDir = os.path.dirname(logFilePath)
        if not os.path.exists(logDir):
            os.mkdir(logDir)

        try:
            if logFileMode.lower() == 'w':
                self.fh = FileHandler(logFilePath, mode='w')
            elif logFileMode.lower() == 'a':
                self.fh = TimedRotatingFileHandler(
                    logFilePath, when='midnight', utc=True)
            else:
                raise ValueError('logger mode {0} not recognised'.format(logFileMode))
        except (IOError, OSError) as ee:
            warnings.warn('log file {0!r} could not be opened for writing: '
                          '{1}'.format(logFilePath, ee, RuntimeWarning))
        else:
            self.fh.setFormatter(fmt)
            self.addHandler(self.fh)

        self.logFilename = logFilePath
        warnings.showwarning = self._showwarning

    def setVerbose(self, verbose):

        # If verbose is set to None or False, changes the logging level of the
        # stdout handler.
        if verbose is False:
            self.handlers[0].setLevel('WARNING')
        elif verbose is None:
            self.handlers[0].setLevel('ERROR')
        else:
            self.handlers[0].setLevel('INFO')

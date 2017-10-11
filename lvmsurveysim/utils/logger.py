#!/usr/bin/env python
# encoding: utf-8
#
# logger.py
#
# Created by José Sánchez-Gallego on 17 Sep 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import click
import datetime
import json
import logging
import os
import pathlib
import re
import shutil
import traceback
import sys
import warnings

from logging.handlers import TimedRotatingFileHandler
# from textwrap import TextWrapper

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter

from twisted.logger import formatEvent, globalLogPublisher

from . import config


# Adds custom log level for print and twisted messages
PRINT = 15
logging.addLevelName(PRINT, 'PRINT')

TWISTED = 12
logging.addLevelName(TWISTED, 'TWISTED')


def print_log_level(self, message, *args, **kws):
    self._log(PRINT, message, args, **kws)


def twisted_log_level(self, message, *args, **kws):
    self._log(TWISTED, message, args, **kws)


logging.Logger._print = print_log_level
logging.Logger._twisted = twisted_log_level


def print_exception_formatted(type, value, tb):
    """A custom hook for printing tracebacks with colours."""

    tbtext = ''.join(traceback.format_exception(type, value, tb))
    lexer = get_lexer_by_name('pytb', stripall=True)
    formatter = TerminalFormatter()
    sys.stderr.write(highlight(tbtext, lexer, formatter))


def colored_formatter(record):
    """Prints log messages with colours."""

    colours = {'info': ('blue', 'normal'),
               'debug': ('magenta', 'normal'),
               'warning': ('yellow', 'normal'),
               'print': ('green', 'normal'),
               'twisted': ('white', 'bold'),
               'error': ('red', 'bold')}

    levelname = record.levelname.lower()

    if levelname == 'error':
        return

    if levelname.lower() in colours:
        levelname_color = colours[levelname][0]
        bold = True if colours[levelname][1] == 'bold' else False
        header = click.style('[{}]: '.format(levelname.upper()), levelname_color, bold=bold)

    message = '{0}'.format(record.msg)

    warning_category = re.match('^(\w+Warning\:).*', message)
    if warning_category is not None:
        warning_category_colour = click.style(warning_category.groups()[0], 'cyan')
        message = message.replace(warning_category.groups()[0], warning_category_colour)

    sub_level = re.match('(\[.+\]:)(.*)', message)
    if sub_level is not None:
        sub_level_name = click.style(sub_level.groups()[0], 'red')
        message = '{}{}'.format(sub_level_name, ''.join(sub_level.groups()[1:]))

    # if len(message) > 79:
    #     tw = TextWrapper()
    #     tw.width = 79
    #     tw.subsequent_indent = ' ' * (len(record.levelname) + 2)
    #     tw.break_on_hyphens = False
    #     message = '\n'.join(tw.wrap(message))

    sys.__stdout__.write('{}{}\n'.format(header, message))
    sys.__stdout__.flush()

    return


class MyFormatter(logging.Formatter):

    warning_fmp = '%(asctime)s - %(levelname)s: %(message)s [%(origin)s]'
    info_fmt = '%(asctime)s - %(levelname)s - %(message)s [%(funcName)s @ ' + \
        '%(filename)s]'

    ansi_escape = re.compile(r'\x1b[^m]*m')

    def __init__(self, fmt='%(levelname)s - %(message)s [%(funcName)s @ ' +
                 '%(filename)s]'):
        logging.Formatter.__init__(self, fmt, datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.getLevelName('PRINT'):
            self._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.getLevelName('TWISTED'):
            self._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.INFO:
            self._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.WARNING:
            self._fmt = MyFormatter.warning_fmp

        record.msg = self.ansi_escape.sub('', record.msg)

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result


Logger = logging.getLoggerClass()
fmt = MyFormatter()


class LoggerStdout(object):
    """A pipe for stdout to a logger."""

    def __init__(self, level):
        self.level = level

    def write(self, message):

        if message != '\n':
            self.level(message)

    def flush(self):
        pass


class MyLogger(Logger):
    """This class is used to set up the logging system.

    The main functionality added by this class over the built-in
    logging.Logger class is the ability to keep track of the origin of the
    messages, the ability to enable logging of warnings.warn calls and
    exceptions, and the addition of colorized output and context managers to
    easily capture messages to a file or list.

    """

    INFO = 15

    # The default actor to log to. It is set by the set_actor() method.
    _actor = None

    def save_log(self, path):
        shutil.copyfile(self.log_filename, os.path.expanduser(path))

    def _show_warning(self, *args, **kwargs):

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

    def _catch_exceptions(self, exctype, value, tb):
        """Catches all exceptions and logs them."""

        # Now we log it.
        self.error('Uncaught exception', exc_info=(exctype, value, tb))

        # First, we print to stdout with some colouring.
        print_exception_formatted(exctype, value, tb)

    def _set_defaults(self, log_level=logging.INFO, redirect_stdout=False):
        """Reset logger to its initial state."""

        # Remove all previous handlers
        for handler in self.handlers[:]:
            self.removeHandler(handler)

        # Set levels
        self.setLevel(logging.DEBUG)

        # Set up the stdout handler
        self.fh = None
        self.sh = logging.StreamHandler()
        self.sh.emit = colored_formatter
        self.addHandler(self.sh)

        self.sh.setLevel(log_level)

        warnings.showwarning = self._show_warning

        # Redirects all stdout to the logger
        if redirect_stdout:
            sys.stdout = LoggerStdout(self._print)

        # Catches exceptions
        sys.excepthook = self._catch_exceptions

    def start_file_logger(self, name='guider', log_file_level=logging.DEBUG,
                          log_file_path=config['logging']['logdir']):
        """Start file logging."""

        log_file_path = pathlib.Path(log_file_path).expanduser() / '{}.log'.format(name)
        logdir = log_file_path.parent

        try:
            logdir.mkdir(parents=True, exist_ok=True)

            # If the log file exists, backs it up before creating a new file handler
            if log_file_path.exists():
                strtime = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H:%M:%S')
                shutil.move(str(log_file_path), str(log_file_path) + '.' + strtime)

            self.fh = TimedRotatingFileHandler(str(log_file_path), when='midnight', utc=True)
            self.fh.suffix = '%Y-%m-%d_%H:%M:%S'
        except (IOError, OSError) as ee:
            warnings.warn('log file {0!r} could not be opened for writing: '
                          '{1}'.format(log_file_path, ee), RuntimeWarning)
        else:
            self.fh.setFormatter(fmt)
            self.addHandler(self.fh)
            self.fh.setLevel(log_file_level)

        self.log_filename = log_file_path

    def set_actor(self, value):
        """Sets the default actor to which to log."""

        self._actor = value

    def debug(self, record, actor=None, **kwargs):
        """Logs a debug message, and writes to the actor users."""

        # If actor=False, does not write to the actor. Otherwise, chooses
        # between the actor argument an the default actor for the logger
        actor = (actor or self._actor) if actor is not False else False

        super(MyLogger, self).debug(record, **kwargs)

        if actor:
            actor.writeToUsers('d', 'text={}'.format(json.dumps(str(record))))

    def info(self, record, actor=None, **kwargs):
        """Logs a info message, and writes to the actor users."""

        actor = (actor or self._actor) if actor is not False else False

        super(MyLogger, self).info(record, **kwargs)

        if actor:
            actor.writeToUsers('i', 'text={}'.format(json.dumps(str(record))))

    def warning(self, record, actor=None, **kwargs):
        """Logs a warning message, and writes to the actor users."""

        actor = (actor or self._actor) if actor is not False else False

        kwargs['extra'] = {'origin': 'actor warning'}
        super(MyLogger, self).warning(record, **kwargs)

        if actor:
            actor.writeToUsers('w', 'text={}'.format(json.dumps(str(record))))


logging.setLoggerClass(MyLogger)
log = logging.getLogger(__name__)
log._set_defaults()  # Inits sh handler


# Creates a twisted observer to redirect messages and failures
def twisted_analyze_event(event):

    text = formatEvent(event)

    if text is None:
        text = ''

    if 'log_failure' in event:
        try:
            traceback = event['log_failure'].getTraceback()
        except Exception:
            traceback = '(UNABLE TO OBTAIN TRACEBACK FROM EVENT)\n'
        text = '\n'.join((text, traceback))
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        log.exception(text)
    else:
        log._twisted(text)


globalLogPublisher.addObserver(twisted_analyze_event)

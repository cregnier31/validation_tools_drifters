import logging
from colorlog import ColoredFormatter
import datetime
import hashlib
import sys
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
import logging.handlers
## Log wrapper
# Creation C.REGNIER Juillet 2014
# Creation C.REGNIER October 2020 :  add color logger

class MPLogger:
    """
    Multiprocessing-safe logger with numeric log level mapping.
    """

    LEVELS = {
        0: logging.NOTSET,
        1: logging.DEBUG,
        2: logging.INFO,
        3: logging.WARNING,
        4: logging.ERROR,
        5: logging.CRITICAL,
        6: logging.DEBUG,   # alias
        7: logging.INFO,    # alias
    }

    def __init__(self, logfile: str, level: int = 2):
        self.logfile = logfile
        self.loglevel = self.LEVELS.get(level, logging.INFO)
        self.queue = multiprocessing.Queue(-1)
        self.listener = None

    def start_listener(self):
        handler = logging.FileHandler(self.logfile, mode="a")
        formatter = logging.Formatter(
            "%(asctime)s  %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        root = logging.getLogger()
        root.setLevel(self.loglevel)
        root.handlers = []
        root.addHandler(handler)

        self.listener = multiprocessing.Process(
            target=self._listener_process, args=(self.queue, handler)
        )
        self.listener.start()

    def _listener_process(self, queue, handler):
        while True:
            try:
                record = queue.get()
                if record is None:
                    break
                logger = logging.getLogger(record.name)
                logger.handle(record)
            except Exception:
                import traceback
                print("Logging error:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
        handler.close()

    def get_logger(self, name=None):
        qh = logging.handlers.QueueHandler(self.queue)
        logger = logging.getLogger(name)
        logger.setLevel(self.loglevel)
        logger.handlers = []
        logger.addHandler(qh)
        logger.propagate = False
        return logger

    def stop_listener(self):
        self.queue.put(None)
        self.listener.join()


class Logger_multi(object):
    """ Multiprocessing-safe Logger with optional logfile or stream """

    def __init__(self, name, stream='', dirlog=''):
        self.name = name
        self.dirlog = dirlog
        current_datetime = datetime.datetime.now()
        log_name = current_datetime.strftime(f"Log_{name}_%Y-%m-%d_%H-%M")
        hashval = hashlib.md5(log_name.encode()).hexdigest()
        self.logname = f'{self.dirlog}{log_name}_{hashval}.txt'

        # Define colored formatter
        LOGFORMAT = " %(log_color)s %(asctime)s  %(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
        self.formatter = ColoredFormatter(
            LOGFORMAT,
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            },
        )

        # Select stream type
        if stream == 'logfile':
            print(f'Add logfile {self.logname}')
            self.stream = logging.FileHandler(self.logname)
        else:
            self.stream = logging.StreamHandler()

        # Setup multiprocessing queue
        self.log_queue = multiprocessing.Queue()
        self.queue_listener = None

    def run(self, level):
        log = logging.getLogger(self.name)
        if log.hasHandlers():
            log.handlers.clear()

        # Configure logger level
        levels = {
            0: logging.NOTSET,
            1: logging.DEBUG,
            2: logging.INFO,
            3: logging.WARNING,
            4: logging.ERROR,
            5: logging.CRITICAL,
            6: logging.DEBUG,
            7: logging.INFO,
        }
        log.setLevel(levels.get(level, logging.INFO))

        # Queue handler for workers
        queue_handler = QueueHandler(self.log_queue)
        log.addHandler(queue_handler)

        # Final handler (main process only)
        self.stream.setLevel(level)
        self.stream.setFormatter(self.formatter)
        self.queue_listener = QueueListener(self.log_queue, self.stream)
        self.queue_listener.start()

        return log

    def stop(self):
        """ Stop the listener cleanly (call at program exit) """
        if self.queue_listener:
            self.queue_listener.stop()


class Logger(object):

    """ Class Logger : define level of log with a number """

    def __init__(self, name, stream='', dirlog='') :
        self.name=name
        self.dirlog = dirlog
        current_datetime = datetime.datetime.now()
        log_name = current_datetime.strftime(f"Log_{name}_%Y-%m-%d_%H-%M")
        # Calculate the MD5 hash of the log name
        hash = hashlib.md5(log_name.encode()).hexdigest()
        self.logname = f'{self.dirlog}{log_name}_{hash}.txt'
        LOGFORMAT = " %(log_color)s %(asctime)s  %(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
        self.formatter = ColoredFormatter(LOGFORMAT,
                         datefmt=None,
                         reset=True,
                         log_colors={
                                     'DEBUG':    'cyan',
                                     'INFO':     'green',
                                     'WARNING':  'yellow',
                                     'ERROR':    'red',
                                     'CRITICAL': 'red,bg_white',
                                      },
                )
        self.stream = stream
        if stream == 'logfile':
            print(f'Add logfile {self.logname}')
            self.stream = logging.FileHandler(self.logname)
        else:
            self.stream = logging.StreamHandler()
        self.handlers = []

    def run(self,level):
        log = logging.getLogger(self.name)
        ## Clean logger
        if (log.hasHandlers()):
           log.handlers.clear()
        ## Level of logging output
        if level == 0 :
            log.setLevel(logging.NOTSET)
        elif level == 1 :
            log.setLevel(logging.DEBUG)
        elif level == 2 :
            log.setLevel(logging.INFO)
        elif level == 3 :
            log.setLevel(logging.WARNING)
        elif level == 4 :
            log.setLevel(logging.ERROR)
        elif level == 5 :
            log.setLevel(logging.CRITICAL)
        elif level == 6 :
            log.setLevel(logging.DEBUG)
            self.stream = ''
        elif level == 7 :
            log.setLevel(logging.INFO)
            self.stream = ''

        # Always attach console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(self.formatter)
        log.addHandler(console_handler) 
        self.handlers.append(console_handler)

        if self.stream == 'logfile':
            file_handler = logging.FileHandler(self.logname)
            file_handler.setLevel(level)
            file_handler.setFormatter(self.formatter)
            log.addHandler(file_handler)
            self.handlers.append(file_handler)

           # self.stream = logging.FileHandler(self.logname)
           # file_handler.setLevel(level)
           # file_handler.setFormatter(self.formatter)
           # log.addHandler(file_handler)
           # self.handlers.append(console_handler)
        #else:
        #    self.stream = logging.StreamHandler()
        #    
        #self.stream.setLevel(level)
        #self.stream.setFormatter(self.formatter)
        #log.addHandler(self.stream)

        return log

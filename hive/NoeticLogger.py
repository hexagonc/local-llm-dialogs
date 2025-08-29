from .HiveUtils import get_standard_datetime_str_from_epoch_for_filenames
from datetime import  datetime
import logging



DEBUG_LOGGER=  "DEBUG_LOGGER"
VERBOSE_LOGGER = "VERBOSE_LOGGER"
INFO_LOGGER = "INFO_LOGGER"
ERROR_LOGGER = "ERROR_LOGGER"
WARNING_LOGGER = "WARNING_LOGGER"

class NoeticLogger:
    def __init__(self, logger_base_name:str = None, log_base_path:str = ""):
        self.loggers = {}

        self.logger_base_name = logger_base_name
        self.log_base_path = log_base_path

    def get_logger(self, log_key:str):
        if log_key not in self.loggers:
            if log_key == DEBUG_LOGGER:
                self.loggers[log_key] = self.setup_logging(self.logger_base_name, log_key)
            elif log_key == VERBOSE_LOGGER:
                self.loggers[log_key] = self.setup_logging(self.logger_base_name,  log_key)
            elif log_key == INFO_LOGGER:
                self.loggers[log_key] = self.setup_logging(self.logger_base_name,  log_key)
            elif log_key == ERROR_LOGGER:
                self.loggers[log_key] = self.setup_logging(self.logger_base_name,  log_key)
            elif log_key == WARNING_LOGGER:
                self.loggers[log_key] = self.setup_logging(self.logger_base_name,  log_key)
            else:
                raise Exception(f"Illegal log level: {log_key}")
        return self.loggers[log_key]

    def setup_logging(self, logger_base_name, log_key:str = INFO_LOGGER) -> logging.Logger:
        now = datetime.now()
        formatted = now.strftime("%Y_%m_%d__%H_%M_%S")
        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        verbose = False
        # Create a logger for each model
        logger = logging.getLogger(logger_base_name)

        # Set the logger level based on the debug flag
        level = logging.INFO
        if log_key == DEBUG_LOGGER:
            level = logging.DEBUG
        elif log_key == VERBOSE_LOGGER:
            verbose = True
        elif log_key == INFO_LOGGER:
            level = logging.INFO
        elif log_key == ERROR_LOGGER:
            level = logging.ERROR
        elif log_key == WARNING_LOGGER:
            level = logging.WARNING
        else:
            raise Exception(f"Illegal log level: {log_key}")
        logger.setLevel(level)
        # Create a file handler that logs debug and higher level messages
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        else:
            log_filename = f"{self.log_base_path}{logger_base_name}_{formatted}.txt"
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger
    def logDebug(self, tag:str, message:str, verbose = None):
        logger:logging.Logger = self.get_logger(DEBUG_LOGGER)
        logger.debug(f"[{tag}] [{message}]")
        if verbose:
            self.logVerbose(tag, message)

    def logInfo(self, tag:str, message:str, verbose = None):
        logger:logging.Logger = self.get_logger(INFO_LOGGER)
        logger.info(f"[{tag}] [{message}]")
        if verbose:
            self.logVerbose(tag, message)

    def logWarning(self, tag:str, message:str, verbose = None):
        logger:logging.Logger = self.get_logger(WARNING_LOGGER)
        logger.warning(f"[{tag}] [{message}]")
        if verbose:
            self.logVerbose(tag, message)

    def logError(self, tag:str, message:str, verbose = None):
        logger:logging.Logger = self.get_logger(ERROR_LOGGER)
        logger.error(f"[{tag}] [{message}]")
        if verbose:
            self.logVerbose(tag, message)

    def logVerbose(self, tag:str, message:str):
        logger:logging.Logger = self.get_logger(VERBOSE_LOGGER)
        logger.info(f"[{tag}] [{message}]")

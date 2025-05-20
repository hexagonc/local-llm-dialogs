
from LLMTools import get_delimited_text
from plisp.Value import Value
from plisp.LispTools import LispTools
from datetime import datetime, timezone
import time


def build_prompt_from_template(env, prompt_template, start_delimiter = "{{", end_delimiter = "}}"):
    replacements = get_delimited_text(prompt_template, start_delimiter, end_delimiter)
    if len(replacements) > 0:
        for field in replacements:
            lisp_exp = field[0]
            res: Value = env.evaluate_exp(lisp_exp)
            prompt_template = prompt_template.replace(f"{start_delimiter}{lisp_exp}{end_delimiter}", res.string() if res.is_string() else res.serialize())
    return prompt_template


def to_svalue(s) -> str:
    return LispTools.make_str(s)


def get_standard_datetime_str_from_epoch_for_filenames(epoch_milli = None):
    if epoch_milli is None:
        epoch_seconds = time.time()
    else:
        epoch_seconds = epoch_milli / 1000
    dt = datetime.fromtimestamp(epoch_seconds)
    return dt.strftime("%a_%B_%d_%Y_%H_%M_%S")

def get_standard_datetime_str_from_epoch_for_logging(epoch_milli = None):
    if epoch_milli is None:
        epoch_seconds = time.time()
    else:
        epoch_seconds = epoch_milli / 1000
    dt = datetime.fromtimestamp(epoch_seconds)
    return dt.strftime("%a_%B_%d_%Y_%H_%M_%S")


def get_standard_datetime_str_from_epoch(epoch_milli = None):
    if epoch_milli is None:
        epoch_seconds = time.time()
    else:
        epoch_seconds = epoch_milli / 1000
    dt = datetime.fromtimestamp(epoch_seconds)
    return dt.strftime("%a %B %d %Y %H:%M:%S")

def parse_epoch_from_standard_datetime_str(date_string, date_format="%a %B %d %Y %H:%M:%S"):
    # Parse the string into a datetime object
    dt = datetime.strptime(date_string, date_format)
    # Convert the datetime object to epoch time in seconds
    epoch_seconds = int(dt.timestamp())
    # Convert seconds to milliseconds
    epoch_milliseconds = epoch_seconds * 1000
    return epoch_milliseconds

def get_file_datetime(epoch_milli = None):
    if epoch_milli is None:
        epoch_seconds = time.time()
    else:
        epoch_seconds = epoch_milli / 1000
    dt = datetime.fromtimestamp(epoch_seconds)
    return dt.strftime("%Y%m%d%M%S")

def get_iso8601_datetime_str_from_epoch(epoch_milli = None):
    if epoch_milli is None:
        epoch_seconds = time.time()
    else:
        epoch_seconds = epoch_milli / 1000
    # Create a datetime object from the epoch time, including timezone information
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    # Format the datetime object into ISO 8601 format
    iso8601_time = dt.isoformat()
    return iso8601_time
# pyTrade/_arg_validators.py

import re

from chameli.dateutils import market_timings, valid_datetime

from .enums import Periodicity

# Import ohlcutils_logger lazily to avoid circular import
def get_ohlcutils_logger():
    """Get ohlcutils_logger instance to avoid circular imports."""
    from . import ohlcutils_logger
    return ohlcutils_logger

# Constants for regex patterns
DEST_BAR_SIZE_PATTERN = r"\d{1,3}[STHDWMY]"
BAR_START_TIME_PATTERN = r"\d{1,2}min"
TIME_PATTERN = r"\d{1,2}:\d{1,2}:{0,1}\d{1,2}"


def _valid_load_symbol_kwargs(**kwargs):
    valid_fills = ("ffill", "drop")
    valid_labels = ("left", "right")
    valid_boolean = (True, False)

    # Use "NSE" as the default exchange if not provided in kwargs
    exchange = kwargs.get("exchange", "NSE")

    # Dynamically fetch market timings based on the exchange
    market_open_time = market_timings.get(exchange, {}).get("open_time", "09:15")
    market_close_time = market_timings.get(exchange, {}).get("close_time", "15:30")
    timezone = market_timings.get("timezone", "Asia/Kolkata")

    defaults = {
        "start_time": None,
        "end_time": None,
        "days": 100,
        "src": Periodicity.DAILY,
        "fill": "ffill",
        "dest_bar_size": None,
        "bar_start_time_in_min": "15min",
        "exchange": exchange,  # Default exchange
        "market_open_time": market_open_time,
        "market_close_time": market_close_time,
        "tz": timezone,
        "label": "left",
        "stub": False,
        "target_weekday": "Monday",
        "adjust_for_holidays": True,
        "adjustment": "pbd",
        "rolling": False,
    }

    validators = {
        "start_time": lambda value: valid_datetime(value, "%Y-%m-%d")[0],
        "end_time": lambda value: valid_datetime(value, "%Y-%m-%d")[0],
        "days": lambda value: isinstance(value, (int, float)),
        "src": lambda value: value in Periodicity.__members__.values(),
        "fill": lambda value: value in valid_fills,
        "dest_bar_size": lambda value: re.match(DEST_BAR_SIZE_PATTERN, value),
        "bar_start_time_in_min": lambda value: re.match(BAR_START_TIME_PATTERN, value),
        "exchange": lambda value: isinstance(value, str),  # Validator for exchange
        "market_open_time": lambda value: re.match(TIME_PATTERN, value),
        "market_close_time": lambda value: re.match(TIME_PATTERN, value),
        "tz": lambda value: True,
        "label": lambda value: value in valid_labels,
        "stub": lambda value: value in valid_boolean,
        "target_weekday": lambda value: value in [None, "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "adjust_for_holidays": lambda value: value in valid_boolean,
        "adjustment": lambda value: value in ["fbd", "pbd", None],
        "rolling": lambda value: value in valid_boolean,
    }

    vkwargs = {key: {"Default": defaults[key], "Validator": validators[key]} for key in defaults}
    return vkwargs


def _valid_metrics_kwargs():
    defaults = {
        "win_ratio": True,
        "count_trades": True,
        "sharpe": False,
        "sortino": False,
        "var": False,
        "drawdown": False,
    }

    validators = {
        "win_ratio": lambda value: isinstance(value, bool),
        "count_trades": lambda value: isinstance(value, bool),
        "sharpe": lambda value: isinstance(value, bool),
        "sortino": lambda value: isinstance(value, bool),
        "var": lambda value: isinstance(value, bool),
        "drawdown": lambda value: isinstance(value, bool),
    }

    vkwargs = {key: {"Default": defaults[key], "Validator": validators[key]} for key in defaults}
    return vkwargs


def _process_kwargs(kwargs, vkwargs):
    """
    Given a "valid kwargs table" and some kwargs, verify that each key-word
    is valid per the kwargs table, and that the value of the kwarg is the
    correct type.  Fill a configuration dictionary with the default value
    for each kwarg, and then substitute in any values that were provided
    as kwargs and return the configuration dictionary.
    """
    # initialize configuration from valid_kwargs_table:
    config = {}
    for key, value in vkwargs.items():
        config[key] = value["Default"]

    # now validate kwargs, and for any valid kwargs
    #  replace the appropriate value in config:
    for key in kwargs.keys():
        if key not in vkwargs:
            get_ohlcutils_logger().log_error(f'Unrecognized kwarg="{key}"', KeyError(f'Unrecognized kwarg="{key}"'), {
                "unrecognized_key": key,
                "valid_keys": list(vkwargs.keys()),
                "function": "_process_kwargs"
            })
            raise KeyError(f'Unrecognized kwarg="{key}"')
        else:
            value = kwargs[key]
            try:
                valid = vkwargs[key]["Validator"](value)
            except Exception as ex:
                get_ohlcutils_logger().log_error(f'kwarg "{key}" validator raised exception to value: "{value}"', ex, {
                    "key": key,
                    "value": value,
                    "function": "_process_kwargs"
                })
                ex.extra_info = f'kwarg "{key}" validator raised exception to value: "{value}"'
                raise
            if not valid:
                import inspect

                v = inspect.getsource(vkwargs[key]["Validator"]).strip()
                get_ohlcutils_logger().log_error(f'kwarg "{key}" validator returned False for value: "{value}"', TypeError(f'kwarg "{key}" validator returned False for value: "{value}"\n    {v}'), {
                    "key": key,
                    "value": value,
                    "validator_source": v,
                    "function": "_process_kwargs"
                })
                raise TypeError(f'kwarg "{key}" validator returned False for value: "{value}"\n    {v}')

        # ---------------------------------------------------------------
        #  At this point in the loop, if we have not raised an exception,
        #      then kwarg is valid as far as we can tell, therefore,
        #      go ahead and replace the appropriate value in config:

        if key in ["start_time", "end_time"]:
            config[key] = valid_datetime(value, "%Y-%m-%d %H:%M:%S")[0]
        else:
            config[key] = value

    return config

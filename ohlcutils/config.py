import datetime as dt
import logging
import os
from typing import Any, Optional

import yaml

# Singleton instance to ensure configuration is loaded once
_config_instance = None

# Create a minimal logger class to avoid circular import
import logging

class ConfigLogger:
    """Minimal logger for config module to avoid circular imports."""

    def __init__(self):
        self.logger = logging.getLogger("ohlcutils.config")
        self.logger.setLevel(logging.WARNING)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_info(self, message, context=None):
        """Log info message."""
        extra = context or {}
        self.logger.info(message, extra=extra)

    def log_warning(self, message, context=None):
        """Log warning message."""
        extra = context or {}
        self.logger.warning(message, extra=extra)

    def log_error(self, message, error=None, context=None, exc_info=True):
        """Log error message."""
        extra = context or {}
        if error:
            extra["error_type"] = type(error).__name__
            extra["error_message"] = str(error)
        self.logger.error(message, extra=extra, exc_info=exc_info)

_logger_instance = ConfigLogger()

def get_ohlcutils_logger():
    """Get logger instance for config module."""
    return _logger_instance


class Config:
    def __init__(self, default_config_path):
        """
        Initialize the Config class.
        :param default_config_path: Path to the main YAML configuration file.
        """
        self.configs = {}
        self.commission_data = {}  # Preloaded commission data
        self.default_config_path = default_config_path
        self.custom_config_path = os.getenv("OHLCUTILS_CONFIG_PATH", None)
        self.config_path = self.custom_config_path or self.default_config_path
        self.base_dir = (
            os.path.dirname(self.config_path) if self.config_path else None
        )  # Base directory for relative paths

        # Load the configuration file
        if self.config_path:
            self._load_config(self.config_path)

    def _load_config(self, config_file):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        with open(config_file, "r") as file:
            try:
                self.configs = yaml.safe_load(file)
                get_ohlcutils_logger().log_info(f"Configuration loaded from {config_file}", {
                    "config_file": config_file
                })
            except Exception as e:
                get_ohlcutils_logger().log_error("Failed to load configuration", e, {
                    "config_file": config_file
                })
                raise

    def __getitem__(self, key):
        if key not in self.configs:
            raise KeyError(f"Key '{key}' not found in configuration.")
        return self.configs[key]

    def get(self, key, default=None):
        return self.configs.get(key, default)


# Global functions for managing configuration


def load_config(default_config_path, force_reload=False):
    """
    Load the configuration globally.
    :param default_config_path: Path to the main configuration file.
    :param force_reload: If True, forces reloading the configuration even if it is already loaded.
    """
    global _config_instance
    if _config_instance is None or force_reload:
        _config_instance = Config(default_config_path)
        get_ohlcutils_logger().log_info(f"Config loaded from file {_config_instance.config_path}", {
            "config_path": _config_instance.config_path
        })
    else:
        get_ohlcutils_logger().log_info("Configuration is already loaded. Skipping reload.")


def is_config_loaded():
    """
    Check if the configuration is already loaded.
    :return: True if the configuration is loaded, otherwise False.
    """
    return _config_instance is not None


def get_config():
    """
    Retrieve the loaded configuration instance.
    :return: Config instance if loaded, otherwise raises ValueError.
    """
    if not is_config_loaded():
        raise ValueError("Configuration has not been loaded yet.")
    return _config_instance


_EXCHANGE_ALIASES = {
    "N": "NSE",
    "NSE": "NSE",
    "NFO": "NSE",
    "B": "BSE",
    "BSE": "BSE",
    "BFO": "BSE",
}
_FNO_EXCHANGES = {"NFO", "BFO"}


def _market_hours_date(value: Any) -> dt.date:
    if value is None:
        return dt.date.today()
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()[:10]
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid market-hours date: {value}")


def _market_time_or_default(value: Any, default: str) -> str:
    candidate = str(value or default)
    if len(candidate) == 5:
        candidate += ":00"
    try:
        dt.datetime.strptime(candidate, "%H:%M:%S")
        return candidate
    except ValueError:
        return default


def _get_market_session_time(
    time_key: str,
    fallback_key: str,
    default: str,
    exchange: Optional[str] = None,
    market: Optional[str] = None,
    symbol: Optional[str] = None,
    as_of: Any = None,
) -> str:
    cfg = get_config()
    fallback = _market_time_or_default(cfg.get(fallback_key, default), default)
    symbol_key = str(symbol or "").upper()
    exchange_key = str(exchange or "").strip().upper()
    if not exchange_key:
        exchange_key = "BSE" if "SENSEX" in symbol_key else "NSE"
    normalized_exchange = _EXCHANGE_ALIASES.get(exchange_key, exchange_key)
    if market is None:
        normalized_market = (
            "FNO"
            if exchange_key in _FNO_EXCHANGES or "_FUT_" in symbol_key or "_OPT_" in symbol_key
            else "CASH"
        )
    else:
        market_key = str(market).strip().upper()
        normalized_market = "FNO" if market_key in {"FNO", "FO", "DERIVATIVES", "DERIVATIVE"} else market_key

    try:
        target_date = _market_hours_date(as_of)
    except ValueError:
        return fallback

    schedules = cfg.get("market_hours", []) or []
    if isinstance(schedules, dict):
        schedules = schedules.get("schedules", []) or []
    applicable = []
    for schedule in schedules if isinstance(schedules, list) else []:
        if not isinstance(schedule, dict) or not schedule.get("effective_date"):
            continue
        try:
            effective_date = _market_hours_date(schedule["effective_date"])
        except ValueError:
            continue
        if effective_date <= target_date:
            applicable.append((effective_date, schedule))

    for _, schedule in sorted(applicable, key=lambda item: item[0], reverse=True):
        exchanges = schedule.get("exchanges") or {}
        exchange_hours = exchanges.get(normalized_exchange) or exchanges.get("DEFAULT") or {}
        market_hours = exchange_hours.get(normalized_market) or exchange_hours.get("DEFAULT") or {}
        session_time = market_hours.get(time_key) if isinstance(market_hours, dict) else None
        if session_time:
            return _market_time_or_default(session_time, fallback)
    return fallback


def get_market_open_time(
    exchange: Optional[str] = None,
    market: Optional[str] = None,
    symbol: Optional[str] = None,
    as_of: Any = None,
    default: str = "09:15:00",
) -> str:
    return _get_market_session_time(
        "open_time", "market_open_time", default, exchange=exchange, market=market, symbol=symbol, as_of=as_of
    )


def get_market_close_time(
    exchange: Optional[str] = None,
    market: Optional[str] = None,
    symbol: Optional[str] = None,
    as_of: Any = None,
    default: str = "15:30:00",
) -> str:
    return _get_market_session_time(
        "close_time", "market_close_time", default, exchange=exchange, market=market, symbol=symbol, as_of=as_of
    )


def is_within_market_hours(
    timestamp: dt.datetime,
    exchange: Optional[str] = None,
    market: Optional[str] = None,
    symbol: Optional[str] = None,
    market_open_time: Optional[str] = None,
    market_close_time: Optional[str] = None,
) -> bool:
    open_time = _market_time_or_default(
        market_open_time or get_market_open_time(exchange, market, symbol, timestamp), "09:15:00"
    )
    close_time = _market_time_or_default(
        market_close_time or get_market_close_time(exchange, market, symbol, timestamp), "15:30:00"
    )
    open_t = dt.datetime.strptime(open_time, "%H:%M:%S").time()
    close_t = dt.datetime.strptime(close_time, "%H:%M:%S").time()
    return open_t <= timestamp.time() < close_t

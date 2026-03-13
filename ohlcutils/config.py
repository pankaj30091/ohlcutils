import logging
import os

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

# pyTrade/data.py

import datetime as dt
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytz


# Import ohlcutils_logger lazily to avoid circular import
def get_ohlcutils_logger():
    """Get ohlcutils_logger instance to avoid circular imports."""
    from . import ohlcutils_logger

    return ohlcutils_logger


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chameli.dateutils import advance_by_biz_days, holidays, is_business_day, market_timings, valid_datetime
from chameli.interactions import readRDS

from ._arg_validators import _process_kwargs, _valid_load_symbol_kwargs
from .config import get_config
from .enums import Periodicity
from .indicators import create_index_if_missing

pd.set_option("future.no_silent_downcasting", True)


def get_dynamic_config():
    return get_config()


# Load configuration
datapath = get_dynamic_config().get("folders").get("basefolder")
splits = readRDS(
    os.path.join(
        datapath,
        get_dynamic_config().get("folders").get("static_data"),
        get_dynamic_config().get("files").get("splits"),
    )
)
splits["date"] = pd.to_datetime(splits.date).dt.strftime("%Y-%m-%d")
symbolchange = readRDS(
    os.path.join(
        datapath,
        get_dynamic_config().get("folders").get("static_data"),
        get_dynamic_config().get("files").get("symbolchange"),
    )
)
symbolchange["effectivedate"] = symbolchange["effectivedate"].dt.tz_localize("UTC")
symbolchange["effectivedate"] = symbolchange["effectivedate"].dt.tz_convert(
    get_dynamic_config().get("static_timezone", "Asia/Kolkata")
)
symbolchange["effectivedate"] = pd.to_datetime(symbolchange.effectivedate).dt.strftime("%Y-%m-%d")

types_supported = ("STK", "IND", "FUT", "OPT")
types_deriv = ("FUT", "OPT")


def get_linked_symbols(short_symbol: str, complete=False) -> pd.DataFrame:
    """Identify securities that have undergone name change.

    Args:
        short_symbol (str): Short Name of the security.
        complete (bool): If True, finds the latest symbol.

    Returns:
        pd.DataFrame: DataFrame with columns ['starttime','endtime','symbol']
                      containing the starttime and endtime of the security identifier(symbol).
    """
    out = pd.DataFrame()
    end_time = dt.date.today().strftime("%Y-%m-%d")
    if complete:
        while True:
            temp = symbolchange.loc[(symbolchange["oldsymbol"] == short_symbol) & (symbolchange["newsymbol"] != "")]
            if len(temp) == 1:
                short_symbol = temp["newsymbol"].item()
            if len(temp) == 0:
                break

    while True:
        temp = symbolchange.loc[(symbolchange["newsymbol"] == short_symbol) & (symbolchange["oldsymbol"] != "")]
        if len(temp) == 1:
            new_row = {
                "starttime": temp.iloc[0]["effectivedate"],
                "endtime": end_time,
                "symbol": temp.iloc[0]["newsymbol"],
            }
            out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)
            end_time = (dt.datetime.strptime(temp.iloc[0]["effectivedate"], "%Y-%m-%d") - dt.timedelta(1)).strftime(
                "%Y-%m-%d"
            )
            short_symbol = temp.iloc[0]["oldsymbol"]
        if len(temp) == 0:
            new_row = {"starttime": "1970-01-01", "endtime": end_time, "symbol": short_symbol}
            out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)
            break
    return out if isinstance(out, pd.DataFrame) else pd.DataFrame()


def get_split_info(short_symbol: str) -> pd.DataFrame:
    """Provides split and bonus information from the database.

    Args:
        short_symbol (str): Short name of security.

    Returns:
        pd.DataFrame: DataFrame with columns ['date','symbol','oldshares','newshares','purpose']
                      containing the split information for security and all its historical linkages.
    """
    split_info = pd.DataFrame(columns=["date", "symbol", "oldshares", "newshares", "purpose"])
    linked_symbols = get_linked_symbols(short_symbol)
    dfs_to_concat = [splits[splits["symbol"] == symbol].copy() for symbol in linked_symbols["symbol"]]
    if dfs_to_concat:
        split_info = pd.concat(dfs_to_concat, ignore_index=True)
    split_info.sort_values("date", axis=0, ascending=False, inplace=True, ignore_index=True)
    return split_info


def _get_data_path(symbol: str, bar_size: Periodicity, for_date: str) -> str:
    """Provides path to market data file. This is a helper function called in load_symbol to retrieve path(s) of market data (.rds) files.

    Args:
        symbol (str): Long name of symbol.
        bar_size (Periodicity): Size of source bar size.
        for_date (str, optional): Required if seeking intra-day data. Formatted as yyyy-mm-dd. Defaults to None.

    Returns:
        str: Path to the data file.
    """
    symbol_list = symbol.split("_", -1)
    if symbol_list[1] not in types_deriv and bar_size.value == Periodicity.DAILY.value:
        return os.path.join(datapath, bar_size.name.lower(), symbol_list[1].lower(), (symbol + ".rds"))
    elif symbol_list[1] in types_deriv and bar_size.value == Periodicity.DAILY.value:
        return os.path.join(datapath, bar_size.name.lower(), symbol_list[1].lower(), symbol_list[2], (symbol + ".rds"))
    else:
        return os.path.join(
            datapath,
            bar_size.name.lower(),
            symbol_list[1].lower(),
            for_date[0:10],
            (symbol + "_" + for_date[0:10] + ".rds"),
        )


def _split_adjust_market_data(md: pd.DataFrame, src: Periodicity, tz: str) -> pd.DataFrame:
    """Helper function to adjust OHLCV market data for splits and bonuses.

    Args:
        md (pd.DataFrame): DataFrame of market data.
        src (Periodicity): Periodicity of the market data.
        tz (str): Timezone for the market data.

    Returns:
        pd.DataFrame: Returns the original DataFrame with an additional column ['splitadjust'].
    """
    md = create_index_if_missing(md)
    if len(md) == 0:
        md["splitadjust"] = None
    else:
        symbol_vector = md.symbol.iloc[0].split("_", -1)
        split_info = get_split_info(symbol_vector[0])
        if not split_info.empty:
            split_info = split_info.groupby("date", sort=False).apply(
                lambda x: x.select_dtypes(include=[np.number]).prod()
            )
            split_info["splitadjust"] = split_info["newshares"] / split_info["oldshares"]
            split_info["splitadjust"] = split_info["splitadjust"].cumprod()
            tz_locale = pytz.timezone(tz)
            split_info.index = pd.to_datetime(split_info.index, format="%Y-%m-%d").tz_localize(tz_locale)
            if src.value == Periodicity.DAILY.value:
                split_info.index = split_info.index + pd.DateOffset(days=-1)
            md = md.merge(split_info["splitadjust"], how="outer", left_index=True, right_index=True)
        else:
            md["splitadjust"] = 1
        md.index.name = "date"
        md["splitadjust"] = md["splitadjust"].bfill().fillna(1)
        md = md.dropna(thresh=4)
    return md


def is_good_df(df):
    if not isinstance(df, pd.DataFrame):
        return False
    if df.empty:
        return False
    if df.isna().all(axis=None):
        return False
    # Remove columns that are all-NA
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return False
    return True


def load_symbol(symbol: str, **kwargs) -> pd.DataFrame:
    """Load market data OHLCV from datastore.

    Args:
        symbol (str): Long name of symbol as given by symbology.
        **kwargs: Optional parameters for loading symbol data. Allowed parameters include:

            start_time (str, optional): Start time for data range in format "YYYY-MM-DD".
                Defaults to None (calculated based on end_time and days).

            end_time (str, optional): End time for data range in format "YYYY-MM-DD".
                Defaults to None (current date if start_time is None, otherwise start_time + days).

            days (int/float, optional): Number of days to load when start_time is not provided.
                Defaults to 100.

            src (Periodicity, optional): Source periodicity for data.
                Defaults to Periodicity.DAILY.

            fill (str, optional): Fill method for missing data. Must be "ffill" or "drop".
                Defaults to "ffill".

            dest_bar_size (str, optional): Destination bar size for timeframe conversion.
                Must match pattern \d{1,3}[STHDWMY] (e.g., "1T", "5S", "1H", "1D", "1W", "1M", "1Y").
                Defaults to None.

            bar_start_time_in_min (str, optional): Bar start time in minutes.
                Must match pattern \d{1,2}min (e.g., "1min", "5min", "15min", "30min").
                Defaults to "15min".

            exchange (str, optional): Exchange name. Defaults to "NSE".

            market_open_time (str, optional): Market open time in format "HH:MM" or "HH:MM:SS".
                Defaults to "09:15".

            market_close_time (str, optional): Market close time in format "HH:MM" or "HH:MM:SS".
                Defaults to "15:30".

            tz (str, optional): Timezone for data. Defaults to "Asia/Kolkata".

            label (str, optional): Label position for time aggregation. Must be "left" or "right".
                Defaults to "left".

            stub (bool, optional): Stub parameter for incomplete periods. Defaults to False.

            target_weekday (str, optional): Target weekday for weekly aggregation.
                Must be None, "Monday", "Tuesday", "Wednesday", "Thursday", or "Friday".
                Defaults to "Monday".

            adjust_for_holidays (bool, optional): Whether to adjust for exchange holidays.
                Defaults to True.

            adjustment (str, optional): Adjustment method for holidays. Must be "fbd", "pbd", or None.
                Defaults to "pbd".

            rolling (bool, optional): Whether to use rolling aggregation. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing OHLCV values with columns:
            - open, high, low, close, settle: Price data
            - volume, tradecount, delivered, tradedvalue: Volume and trade data
            - symbol: Symbol identifier
            - splitadjust: Split adjustment flag
            - aopen, ahigh, alow, aclose, asettle, avolume: Adjusted price and volume data
            - date: Datetime index with timezone Asia/Kolkata

    Raises:
        KeyError: If unrecognized kwargs are provided.
        ValueError: If kwargs values don't match expected patterns or types.

    Example:
        >>> import ohlcutils
        >>> # Basic usage
        >>> df = ohlcutils.load_symbol("NIFTY_IND___")
        >>>
        >>> # With custom parameters
        >>> df = ohlcutils.load_symbol(
        ...     "NIFTY_IND___",
        ...     start_time="2024-01-01",
        ...     end_time="2024-01-31",
        ...     src=ohlcutils.Periodicity.PERMIN,
        ...     dest_bar_size="1H",
        ...     exchange="NSE",
        ...     fill="ffill"
        ... )
    """
    out = pd.DataFrame(
        columns=[
            "open",
            "high",
            "low",
            "close",
            "settle",
            "volume",
            "tradecount",
            "delivered",
            "tradedvalue",
            "symbol",
            "splitadjust",
            "aopen",
            "ahigh",
            "alow",
            "aclose",
            "asettle",
            "avolume",
        ],
        index=pd.DatetimeIndex([], dtype="datetime64[ns, Asia/Kolkata]", name="date", freq=None),
    )

    params = _process_kwargs(kwargs, _valid_load_symbol_kwargs())
    market_close_time = (
        dt.datetime.strptime(params["market_close_time"], "%H:%M:%S") - dt.timedelta(seconds=1)
    ).strftime("%H:%M:%S")

    # Sanity check - dest_bar_size should be larger than src
    bar_size = {Periodicity.DAILY.value: "1D", Periodicity.PERMIN.value: "1min", Periodicity.PERSECOND.value: "1S"}.get(
        params["src"].value, None
    )

    if bar_size is None:
        return out

    if params["dest_bar_size"] is not None:
        if not any(bs in params["dest_bar_size"] for bs in ["D", "ME", "W", "Y", "min", "S"]):
            return out

    # Generate start_time and end_time if not provided
    if params["end_time"] is None:
        params["end_time"] = (
            dt.datetime.now().strftime("%Y-%m-%d")
            if params["start_time"] is None
            else (valid_datetime(params["start_time"]) + dt.timedelta(days=int(params["days"]))).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    if params["start_time"] is None:
        params["start_time"] = (
            dt.datetime.strptime(params["end_time"][0:10], "%Y-%m-%d") - dt.timedelta(days=params["days"])
        ).strftime("%Y-%m-%d")

    params["start_time"] = valid_datetime(params["start_time"], "%Y-%m-%d %H:%M:%S")[0]
    params["end_time"] = valid_datetime(params["end_time"], "%Y-%m-%d %H:%M:%S")[0]

    # Generate paths to rds file
    split_info = get_linked_symbols(symbol.split("_", 5)[0])
    paths = []
    exchange_holidays = holidays.get(params["exchange"], [])
    if params["src"].value != Periodicity.DAILY.value:
        custom_calendar = pd.tseries.offsets.CustomBusinessDay(holidays=exchange_holidays)
        business_days = pd.date_range(
            dt.date.fromisoformat(params["start_time"][0:10]),
            dt.date.fromisoformat(params["end_time"][0:10]),
            freq=custom_calendar,
        )
        for i in range(len(split_info)):
            bdays = np.where(
                (business_days >= split_info["starttime"][i]) & (business_days <= split_info["endtime"][i])
            )
            bdays = business_days[bdays]
            symbol_temp = symbol.split("_", -1)
            symbol_temp[0] = split_info["symbol"][i]
            symbol_temp_str = "_".join(symbol_temp)
            paths.extend(
                [
                    _get_data_path(symbol_temp_str, params["src"], for_date=dt.datetime.strftime(bd, "%Y-%m-%d"))
                    for bd in bdays
                ]
            )
    else:
        for i in range(len(split_info)):
            if params["start_time"] <= split_info["endtime"][i] and params["end_time"] >= split_info["starttime"][i]:
                symbol_temp = symbol.split("_", -1)
                symbol_temp[0] = split_info["symbol"][i]
                symbol_temp_str = "_".join(symbol_temp)
                paths.extend([_get_data_path(symbol_temp_str, params["src"], params["src"])])

    # Read rds from path
    if len(paths) == 0:
        return pd.DataFrame()
    elif len(paths) == 1:
        try:
            out = readRDS(paths[0])
        except Exception as e:
            get_ohlcutils_logger().log_error("Failed to read RDS file", e, {"file_path": paths[0], "symbol": symbol})
            return pd.DataFrame()
    else:
        list_out = []
        for p in paths:
            try:
                out = readRDS(p)
                list_out.append(out)
            except Exception as e:
                get_ohlcutils_logger().log_error("Failed to read RDS file", e, {"file_path": p, "symbol": symbol})
                warnings.warn(f"Failed to read RDS file {p}: {e}", RuntimeWarning)
        filtered_list_out = [df.dropna(axis=1, how="all") for df in list_out if is_good_df(df)]
        if filtered_list_out:
            out = pd.concat(filtered_list_out, ignore_index=True)
        else:
            out = pd.DataFrame()

    if isinstance(out, pd.DataFrame) and len(out) > 0:
        # Set symbol
        out["symbol"] = symbol

        if "splitadjust" in out.columns:
            out = out.drop("splitadjust", axis=1)

        # Remove duplicates
        out = out[~out.date.duplicated(keep="last")]

        # Set index
        out["date"] = pd.to_datetime(out["date"])
        out["date"] = out["date"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")

        # date_aware_local = [pytz.utc.localize(d).astimezone(params["tz"]) for d in out.loc[:, "date"]]
        # out.loc[:, "date"] = pd.DatetimeIndex(date_aware_local).tz_localize(None)
        out = out.set_index("date")
        out.sort_index(inplace=True)

        # Fill or drop rows
        if params["fill"] == "ffill":
            if bar_size is not None:
                out = out.resample(bar_size, label="left").asfreq()
                out.loc[:, "volume"] = out.loc[:, "volume"].fillna(0)
                out = out.ffill().infer_objects(copy=False)  # Explicitly infer object dtypes
            # Exclude rows mapping to holidays.
            if any(substring in bar_size for substring in ["min", "S"]):
                out = out[pd.to_datetime(out.index).dayofweek < 5]
                out = out[
                    ~pd.to_datetime(out.index)
                    .normalize()
                    .isin(pd.to_datetime(holidays[params["exchange"]]).tz_localize(params["tz"]))
                ]  # Exclude rows mapping to holidays.
                out = out.between_time(
                    params["market_open_time"], market_close_time
                )  # Exclude time outside trading hours
            elif "D" in bar_size:
                out = out[pd.to_datetime(out.index).dayofweek < 5]
                out = out[
                    ~pd.to_datetime(out.index)
                    .normalize()
                    .isin(pd.to_datetime(holidays[params["exchange"]]).tz_localize(params["tz"]))
                ]  # Exclude rows mapping to holidays.
        elif params["fill"] == "drop":
            out.dropna(thresh=4, inplace=True)  # Drop rows where at least 4 OHLCV values are missing
        else:
            pass

        if params["dest_bar_size"] is None:
            out = out.loc[params["start_time"] : params["end_time"]]

        # Split adjust
        out = _split_adjust_market_data(out, params["src"], params["tz"])
        out = out.apply(lambda x: pd.to_numeric(x, errors="coerce") if x.name != "symbol" else x)

        if "open" in out.columns:
            out["aopen"] = out["open"] / out["splitadjust"]
        if "high" in out.columns:
            out["ahigh"] = out["high"] / out["splitadjust"]
        if "low" in out.columns:
            out["alow"] = out["low"] / out["splitadjust"]
        if "close" in out.columns:
            out["aclose"] = out["close"] / out["splitadjust"]
        if "settle" in out.columns:
            out["asettle"] = out["settle"] / out["splitadjust"]
        else:
            out["settle"] = out["close"]
            out["asettle"] = out["settle"] / out["splitadjust"]
        if "volume" in out.columns:
            out["avolume"] = out["volume"] * out["splitadjust"]
        if "delivered" in out.columns:
            out["adelivered"] = out["delivered"] * out["splitadjust"]

    # Convert all columns other than symbol to float
    cols = out.columns.drop(["symbol"])
    out[cols] = out[cols].apply(pd.to_numeric, errors="coerce")

    if params["dest_bar_size"] is not None:
        out = change_timeframe(
            out,
            dest_bar_size=params["dest_bar_size"],
            bar_start_time_in_min=params["bar_start_time_in_min"],
            exchange=params["exchange"],
            label=params["label"],
            fill=params["fill"],
            target_weekday=params["target_weekday"],
            adjust_for_holidays=params["adjust_for_holidays"],
            adjustment=params["adjustment"],
            rolling=params["rolling"],
        )
        if not params["stub"]:
            out = out.loc[params["start_time"] : params["end_time"]]
        elif params["label"] == "right":
            end_time_index = pd.Timestamp(valid_datetime(params["end_time"]), tz="Asia/Kolkata")
            if out.index[-1] > end_time_index:
                as_list = out.index.tolist()
                as_list[-1] = end_time_index
                out.index = as_list
        out.sort_index(axis=0, ascending=True, inplace=True)
    return out


def change_timeframe(
    md: pd.DataFrame,
    dest_bar_size: str,
    bar_start_time_in_min: str = "15min",
    exchange: str = "NSE",
    label: str = "left",
    fill: str = "ffill",
    addl_column_merge_rules: dict = {},
    target_weekday: str = "Monday",
    adjust_for_holidays: bool = True,
    adjustment="fbd",
    rolling=False,
) -> pd.DataFrame:
    """
    Resample market data to a different timeframe.

    This function adjusts the timeframe of a given market data DataFrame (`md`) to a specified
    destination bar size (`dest_bar_size`). It supports both intraday (e.g., '15min', '1H') and
    higher timeframes (e.g., '1D', '1W', '1M'). The function also handles holidays, non-trading
    hours, and missing data based on the provided parameters.

    Args:
        md (pd.DataFrame): Input market data DataFrame with a datetime index.
        dest_bar_size (str): Target bar size (e.g., '15min', '1H', '1D', '1W', '1M').
        bar_start_time_in_min (str, optional): Baseline timestamp to start the bar (e.g., '15min'). Defaults to '15min'.
        exchange (str, optional): Exchange identifier to fetch market timings and holidays. Defaults to 'NSE'.
        label (str, optional): Determines the timestamp placement for the resampled bar:
            - 'left': Timestamp at the start of the bar.
            - 'right': Timestamp at the end of the bar.
            Defaults to 'left'.
        fill (str, optional): Handling of missing data:
            - 'ffill': Forward-fill missing rows.
            - 'drop': Drop rows with missing data.
            Defaults to 'ffill'.
        addl_column_merge_rules (dict, optional): Custom aggregation rules for additional columns. Defaults to {}.
        target_weekday (str, optional): Target weekday for aligning weekly bars (e.g., 'Monday'). Only applicable for weekly bars. Defaults to 'Monday'.
        adjust_for_holidays (bool, optional): If True, adjusts resampled dates to avoid holidays. Defaults to True.
        adjustment (str, optional): Adjustment method for holidays:
            - 'fbd': Forward business day.
            - 'pbd': Previous business day.
            Applicable only if `adjust_for_holidays` is True. Defaults to 'fbd'.

    Returns:
        pd.DataFrame: Resampled market data DataFrame with the specified timeframe.

    Raises:
        ValueError: If `dest_bar_size` is invalid.
        ValueError: If `label` is not 'left' or 'right'.
        ValueError: If `target_weekday` is invalid for non-weekly bars.

    Key Features:
        1. **Intraday Resampling**:
           - Supports bar sizes like '15min', '1H', etc.
           - Filters out non-trading hours and holidays.

        2. **Higher Timeframe Resampling**:
           - Supports bar sizes like '1D', '1W', '1M', '1Y'.
           - Aligns weekly bars to a specific weekday if `target_weekday` is provided.
           - Adjusts for holidays if `adjust_for_holidays` is True.

        3. **Custom Aggregation Rules**:
           - Default rules for OHLCV columns:
             - 'open': First value.
             - 'high': Maximum value.
             - 'low': Minimum value.
             - 'close': Last value.
             - 'volume': Sum of values.
           - Additional rules can be specified via `addl_column_merge_rules`.

        4. **Timezone Handling**:
           - Ensures the output DataFrame retains the timezone of the input data.

        5. **Missing Data Handling**:
           - Forward-fill or drop missing rows based on the `fill` parameter.

    Example Usage:
        ```python
        # Resample market data to 1-hour bars
        resampled_data = change_timeframe(
            md=market_data,
            dest_bar_size="1H",
            exchange="NSE",
            label="left",
            fill="ffill"
        )

        # Resample to weekly bars aligned to Tuesday
        weekly_data = change_timeframe(
            md=market_data,
            dest_bar_size="1W",
            target_weekday="Tuesday",
            adjust_for_holidays=True
        )
        ```
    """
    # Handle empty DataFrame
    if md.empty:
        return md
    md = create_index_if_missing(md)
    # --- rolling logic ---
    if rolling:
        if label == "left" and rolling:
            get_ohlcutils_logger().log_warning(
                "Rolling bars are not supported with label='left'. Defaulting to label='right'.",
                {"original_label": "left", "new_label": "right", "rolling": rolling},
            )
            label = "right"
        rolling_days = int(dest_bar_size.replace("D", "")) if "D" in dest_bar_size else 1
        all_bars = []
        for offset in range(rolling_days):
            if offset == 0:
                df_offset = md
            else:
                df_offset = md.iloc[:-offset]
            bars = change_timeframe(
                df_offset,
                dest_bar_size=dest_bar_size,
                bar_start_time_in_min=bar_start_time_in_min,
                exchange=exchange,
                label="right",
                fill=fill,
                addl_column_merge_rules=addl_column_merge_rules,
                target_weekday=target_weekday,
                adjust_for_holidays=adjust_for_holidays,
                adjustment=adjustment,
                rolling=False,  # prevent recursion
            )
            all_bars.append(bars)
        merged = pd.concat(all_bars).sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]
        return merged

    # Validate `dest_bar_size`
    valid_frequencies = ["S", "min", "H", "D", "W", "ME", "Y"]
    if not any(freq in dest_bar_size for freq in valid_frequencies):
        get_ohlcutils_logger().log_error(
            f"Invalid `dest_bar_size`: {dest_bar_size}",
            ValueError(f"Invalid `dest_bar_size`: {dest_bar_size}"),
            {"dest_bar_size": dest_bar_size, "valid_frequencies": valid_frequencies},
        )
        raise ValueError(f"Invalid `dest_bar_size`: {dest_bar_size}")

    # Validate `label`
    if label not in ["left", "right"]:
        get_ohlcutils_logger().log_error(
            f"Invalid `label`: {label}. Must be 'left' or 'right'.",
            ValueError(f"Invalid `label`: {label}. Must be 'left' or 'right'."),
            {"label": label, "valid_labels": ["left", "right"]},
        )
        raise ValueError(f"Invalid `label`: {label}. Must be 'left' or 'right'.")

    # Validate `target_weekday`
    if target_weekday:
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if target_weekday not in weekdays:
            get_ohlcutils_logger().log_error(
                f"Invalid `target_weekday`: {target_weekday}. Must be a valid weekday name.",
                ValueError(f"Invalid `target_weekday`: {target_weekday}. Must be a valid weekday name."),
                {"target_weekday": target_weekday, "valid_weekdays": weekdays},
            )
            raise ValueError(f"Invalid `target_weekday`: {target_weekday}. Must be a valid weekday name.")

    # Fetch market timings and timezone for the exchange
    market_open_time = market_timings.get(exchange, {}).get("open_time", "09:15")
    market_close_time = market_timings.get(exchange, {}).get("close_time", "15:30")
    tz = market_timings.get(exchange, {}).get("timezone", "Asia/Kolkata")

    # Adjust market close time to exclude the last second
    market_close_time = (dt.datetime.strptime(market_close_time, "%H:%M:%S") - dt.timedelta(seconds=1)).strftime(
        "%H:%M:%S"
    )

    # Define aggregation rules for resampling
    agg_columns = {}
    for column in md:
        if "open" in column:
            agg_columns[column] = "first"
        elif "high" in column:
            agg_columns[column] = "max"
        elif "low" in column:
            agg_columns[column] = "min"
        elif "settle" in column or "close" in column:
            agg_columns[column] = "last"
        elif "volume" in column or "delivered" in column or "tradedvalue" in column:
            agg_columns[column] = "sum"
        else:
            agg_columns[column] = "first"
    agg_columns.update(addl_column_merge_rules)
    # md.index = md.index.tz_localize("Asia/Kolkata", ambiguous="infer")
    if "W" in dest_bar_size:
        weekday_to_pandas_freq = {
            "Monday": "W-SUN",
            "Tuesday": "W-MON",
            "Wednesday": "W-TUE",
            "Thursday": "W-WED",
            "Friday": "W-THU",
            "Saturday": "W-FRI",
            "Sunday": "W-SAT",
        }
        # Dynamically determine the offset from the timezone of md.index
        weekly_freq = weekly_freq = (
            f"{dest_bar_size[:1]}{weekday_to_pandas_freq[target_weekday]}"  # e.g., "1W-MON" for Tuesday target
        )
        md = md.resample(weekly_freq, label=label).agg(agg_columns).ffill()
        md.index = md.index + pd.Timedelta(days=1)
    elif any(freq in dest_bar_size for freq in ["ME", "Y"]):
        md = md.resample(dest_bar_size, label=label).agg(agg_columns).ffill()
        if label == "left":
            md.index = md.index + pd.DateOffset(days=1)
    elif any(freq in dest_bar_size for freq in ["D"]):  # Handle arbitrary business day-based sampling
        # Extract the number of business days from dest_bar_size (e.g., "5D" -> 5)
        period_length = int(dest_bar_size.replace("D", ""))

        # Define a custom business day frequency that excludes weekends and holidays
        business_days = pd.offsets.CustomBusinessDay(holidays=holidays.get(exchange, []))

        # Generate snapshot dates with exactly `period_length` business days between them
        snapshot_dates = pd.date_range(
            end=md.index[0].normalize(),
            start=(md.index[-1] + dt.timedelta(days=1)).normalize(),
            freq=-1 * business_days * period_length,
            inclusive="left",
        )
        # Ensure the first day in md.index is included in the snapshot dates
        if md.index[0] not in snapshot_dates:
            snapshot_dates = snapshot_dates.append(pd.DatetimeIndex([md.index[0]]))
        snapshot_dates = snapshot_dates[::-1]  # reverse dates so that earliest is first

        snapshot_dates = pd.to_datetime(snapshot_dates)
        snapshot_dates = snapshot_dates.sort_values()

        # Create intervals
        intervals = pd.IntervalIndex.from_breaks(snapshot_dates, closed="left")

        # Assign interval label to each row
        md["interval"] = pd.cut(md.index, bins=intervals)

        # Group by interval and aggregate
        aggregated = md.groupby("interval", observed=False).agg(agg_columns)

        # Assign a representative timestamp to each interval based on the label
        if not aggregated.empty and aggregated.index is not None:  # Ensure the DataFrame is not empty and has an index
            business_day_offset = pd.offsets.CustomBusinessDay(holidays=holidays.get(exchange, []))
            if label == "left":
                aggregated.index = [interval.left for interval in aggregated.index]
            elif label == "right":
                aggregated.index = [interval.right - business_day_offset for interval in aggregated.index]
            else:
                get_ohlcutils_logger().log_error(
                    f"Invalid `label`: {label}. Must be 'left' or 'right'.",
                    ValueError(f"Invalid `label`: {label}. Must be 'left' or 'right'."),
                    {"label": label, "valid_labels": ["left", "right"]},
                )
                raise ValueError(f"Invalid `label`: {label}. Must be 'left' or 'right'.")
            aggregated.index.name = "date"  # Safely assign the name to the index

        md = aggregated.copy()
        # Normalize the index to ensure the time part is set to 00:00:00
        md.index = md.index.normalize()

    else:
        md = (
            md.resample(dest_bar_size, offset=pd.Timedelta(bar_start_time_in_min), label=label).agg(agg_columns).ffill()
        )

    # Normalize the index to ensure the time part is set to 00:00:00
    if any(barsize in dest_bar_size for barsize in ["D", "W", "ME", "Y"]):
        md.index = md.index.normalize()

    # Restore timezone information
    if md.index.tz is None:
        md.index = md.index.tz_localize(tz)

    # Filter out holidays and non-trading hours
    if any(substring in dest_bar_size for substring in ["min", "H", "S"]):
        md = md[md.index.dayofweek < 5]  # Exclude weekends
        md = md[
            ~md.index.normalize().isin(pd.to_datetime(holidays.get(exchange, [])).tz_localize(tz))
        ]  # Exclude holidays
        md = md.between_time(market_open_time, market_close_time)  # Exclude non-trading hours
    elif "D" in dest_bar_size:
        md = md[
            ~md.index.normalize().isin(pd.to_datetime(holidays.get(exchange, [])).tz_localize(tz))
        ]  # Exclude holidays

    if adjust_for_holidays and any(barsize in dest_bar_size for barsize in ["D", "W", "ME", "Y"]):
        md.index = pd.to_datetime(md.index)
        md.index = md.index.map(
            lambda d: (
                advance_by_biz_days(d, 0, adjustment=adjustment, exchange=exchange)
                if not is_business_day(d, exchange)
                else d
            )
        )

    # Handle missing data
    if fill == "ffill":
        md = md.ffill(axis=0)
    elif fill == "drop":
        md.dropna(thresh=4, inplace=True)

    # Ensure numeric columns are properly converted
    md.index.name = "date"
    cols = md.columns.drop(["symbol"], errors="ignore")
    md[cols] = md[cols].apply(pd.to_numeric, errors="coerce")

    return md

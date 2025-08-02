from typing import Callable, List

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

# Import ohlcutils_logger lazily to avoid circular import
def get_ohlcutils_logger():
    """Get ohlcutils_logger instance to avoid circular imports."""
    from . import ohlcutils_logger
    return ohlcutils_logger


def align_dataframes_on_common_dates(dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Aligns a list of DataFrames to have only common dates in their indices.

    Args:
        dataframes (List[pd.DataFrame]): A list of DataFrames with datetime indices.

    Returns:
        List[pd.DataFrame]: A list of DataFrames with only the common dates.

    Raises:
        ValueError: If there are no common dates across all DataFrames.
    """
    if len(dataframes) < 2:
        get_ohlcutils_logger().log_error("At least two DataFrames are required.", ValueError("At least two DataFrames are required."), {
            "dataframes_count": len(dataframes)
        })
        raise ValueError("At least two DataFrames are required.")

    for i, df in enumerate(dataframes):
        if "date" in df.columns:
            dataframes[i] = df.set_index("date")
        elif not isinstance(df.index, pd.DatetimeIndex):
            get_ohlcutils_logger().log_error("Date column or datetime index is required.", ValueError("Date column or datetime index is required."), {
                "dataframe_index": type(df.index).__name__,
                "has_date_column": "date" in df.columns
            })
            raise ValueError("Date column or datetime index is required.")

    for i, df in enumerate(dataframes):
        dataframes[i] = df.sort_index()

    # Find common dates across all DataFrames
    common_dates = dataframes[0].index
    for df in dataframes[1:]:
        common_dates = common_dates.intersection(df.index)

    if common_dates.empty:
        get_ohlcutils_logger().log_error("No common dates across all DataFrames.", ValueError("No common dates across all DataFrames."), {
            "dataframes_count": len(dataframes)
        })
        raise ValueError("No common dates across all DataFrames.")

    # Align all DataFrames to the common dates
    aligned_dataframes = [df.loc[common_dates].copy() for df in dataframes]

    return aligned_dataframes


def create_index_if_missing(md):
    """Create index if missing in the DataFrame.

    Args:
        md (pd.DataFrame): Market data DataFrame.

    Returns:
        pd.DataFrame: DataFrame with a datetime index.
    """
    if "date" in md.columns:
        md = md.set_index("date")
    elif not isinstance(md.index, pd.DatetimeIndex):
        get_ohlcutils_logger().log_error("Date column or datetime index is required.", ValueError("Date column or datetime index is required."), {
            "dataframe_index": type(md.index).__name__,
            "has_date_column": "date" in md.columns
        })
        raise ValueError("Date column or datetime index is required.")
    md = md.sort_index()
    return md


def calculate_beta(md, md_benchmark, columns: dict = {"close": "asettle"}, window=252):
    """Calculate rolling beta using a moving window or all values if window=None.

    Args:
        md (pd.DataFrame): Market data DataFrame with a datetime index.
        md_benchmark (pd.DataFrame): Benchmark market data DataFrame with a datetime index.
        col (str, optional): Column name to calculate beta on. Defaults to "close".
        window (int or None, optional): Rolling window size. If None, calculate beta using all values. Defaults to 252.

    Returns:
        pd.DataFrame: DataFrame with the same index as `md` and a single column named "beta_series".
                      If window=None, the DataFrame will have a single row with the beta value.
    """
    # Align the two DataFrames on common dates
    [md, md_benchmark] = align_dataframes_on_common_dates([md, md_benchmark])
    required_keys = ["close"]
    if not all(key in columns for key in required_keys):
        get_ohlcutils_logger().log_error(f"The `columns` argument must include mappings for {required_keys}.", ValueError(f"The `columns` argument must include mappings for {required_keys}."), {
            "required_keys": required_keys,
            "provided_keys": list(columns.keys()),
            "missing_keys": [key for key in required_keys if key not in columns],
            "function": "calculate_beta"
        })
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    close_col = columns["close"]

    # Ensure the mapped columns exist in the DataFrame
    if not all(col in md.columns for col in [close_col]):
        get_ohlcutils_logger().log_error("Mapped columns must exist in the DataFrame.", ValueError("Mapped columns must exist in the DataFrame."), {
            "required_columns": [close_col],
            "available_columns": list(md.columns),
            "missing_columns": [col for col in [close_col] if col not in md.columns],
            "function": "calculate_beta"
        })
        raise ValueError("Mapped columns must exist in the DataFrame.")

    # Calculate log returns
    log_ret = np.log(md[close_col] / md[close_col].shift(1))
    log_ret_bench = np.log(md_benchmark[close_col] / md_benchmark[close_col].shift(1))

    if window is None:
        # Calculate beta using all values
        valid_data = pd.DataFrame({"log_ret": log_ret, "log_ret_bench": log_ret_bench}).dropna()
        beta, _, _, _, _ = linregress(valid_data["log_ret_bench"], valid_data["log_ret"])
        beta_series = pd.DataFrame({"beta_series": [beta]}, index=["all_data"])
    else:
        # Calculate rolling standard deviation of the benchmark
        std_bench = log_ret_bench.rolling(window).std()
        std_bench[std_bench == 0] = float("nan")  # Avoid division by zero

        # Calculate rolling beta
        beta_values = log_ret.rolling(window).corr(log_ret_bench) * (log_ret.rolling(window).std() / std_bench)
        beta_series = beta_values.to_frame(name="beta_series")
        # Return as a DataFrame

    return beta_series


def calculate_ratio_bars(
    md,
    md_benchmark,
    columns: dict = {"open": "aopen", "high": "ahigh", "low": "alow", "close": "asettle"},
):
    """
    Calculate ratio-adjusted OHLC bars for a given market data DataFrame.

    This function adjusts the OHLC (Open, High, Low, Close) prices of a given market data DataFrame
    based on the ratio of the asset's prices to a benchmark market data DataFrame. The adjustment
    is done to normalize the asset's prices relative to the benchmark.

    Args:
        md (pd.DataFrame): Market data DataFrame containing OHLC prices and a 'date' column.
        md_benchmark (pd.DataFrame): Benchmark market data DataFrame containing OHLC prices and a 'date' column.
        open_col (str): Column name for the open prices in the market data DataFrame. Default is "aopen".
        high_col (str): Column name for the high prices in the market data DataFrame. Default is "ahigh".
        low_col (str): Column name for the low prices in the market data DataFrame. Default is "alow".
        close_col (str): Column name for the close prices in the market data DataFrame. Default is "asettle".

    Returns:
        pd.DataFrame: DataFrame containing the ratio-adjusted OHLC prices with columns ["ratio_adj_open", "ratio_adj_high", "ratio_adj_low", "ratio_adj_close","symbol"].

    Raises:
        ValueError: If there are no common dates between the market data and the benchmark data.
    """
    [md, md_benchmark] = align_dataframes_on_common_dates([md, md_benchmark])
    required_keys = ["open", "high", "low", "close"]
    if not all(key in columns for key in required_keys):
        get_ohlcutils_logger().log_error(f"The `columns` argument must include mappings for {required_keys}.", ValueError(f"The `columns` argument must include mappings for {required_keys}."), {
            "required_keys": required_keys,
            "provided_keys": list(columns.keys()),
            "missing_keys": [key for key in required_keys if key not in columns],
            "function": "calculate_ratio_bars"
        })
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    open_col = columns["open"]
    high_col = columns["high"]
    low_col = columns["low"]
    close_col = columns["close"]

    # Ensure the mapped columns exist in the DataFrame
    if not all(col in md.columns for col in [open_col, high_col, low_col, close_col]):
        get_ohlcutils_logger().log_error("Mapped columns must exist in the DataFrame.", ValueError("Mapped columns must exist in the DataFrame."), {
            "required_columns": [open_col, high_col, low_col, close_col],
            "available_columns": list(md.columns),
            "missing_columns": [col for col in [open_col, high_col, low_col, close_col] if col not in md.columns],
            "function": "calculate_ratio_bars"
        })
        raise ValueError("Mapped columns must exist in the DataFrame.")

    # Check for zero values in the first open columns to avoid division by zero
    if md.iloc[0][open_col] == 0 or md_benchmark.iloc[0][open_col] == 0:
        get_ohlcutils_logger().log_error("The first open value in md or md_benchmark is zero, cannot calculate ratio bars.", ValueError("The first open value in md or md_benchmark is zero, cannot calculate ratio bars."), {
            "md_first_open": md.iloc[0][open_col],
            "md_benchmark_first_open": md_benchmark.iloc[0][open_col],
            "function": "calculate_ratio_bars"
        })
        raise ValueError("The first open value in md or md_benchmark is zero, cannot calculate ratio bars.")

    # Calculate the scaling factor to make the ratio of the first open values equal to 100
    scaling_factor = 100 / (md.iloc[0][open_col] / md_benchmark.iloc[0][open_col])

    # Calculate ratio bars
    ratio_bars = pd.DataFrame(
        {
            "ratio_adj_open": (md[open_col] / md_benchmark[open_col]) * scaling_factor,
            "ratio_adj_high": (md[high_col] / md_benchmark[high_col]) * scaling_factor,
            "ratio_adj_low": (md[low_col] / md_benchmark[low_col]) * scaling_factor,
            "ratio_adj_close": (md[close_col] / md_benchmark[close_col]) * scaling_factor,
            "symbol": md["symbol"],
        },
        index=md.index,
    )

    return ratio_bars


def calculate_beta_adjusted_bars(
    md: pd.DataFrame,
    md_benchmark: pd.DataFrame,
    beta_days=252,
    columns: dict = {"open": "aopen", "high": "ahigh", "low": "alow", "close": "asettle"},
):
    """
    Calculate beta-adjusted OHLC bars for a given market data DataFrame.
    This function adjusts the OHLC (Open, High, Low, Close) prices of a given market data DataFrame
    based on the beta of the asset relative to a benchmark market data DataFrame. The adjustment
    is done to remove the systematic risk component from the asset's returns.
    Parameters:
    md (pd.DataFrame): Market data DataFrame containing OHLC prices and a 'date' column.
    md_benchmark (pd.DataFrame): Benchmark market data DataFrame containing OHLC prices and a 'date' column.
    open_col (str): Column name for the open prices in the market data DataFrame. Default is "aopen".
    high_col (str): Column name for the high prices in the market data DataFrame. Default is "ahigh".
    low_col (str): Column name for the low prices in the market data DataFrame. Default is "alow".
    close_col (str): Column name for the close prices in the market data DataFrame. Default is "asettle".
    Returns:
    pd.DataFrame: DataFrame containing the beta-adjusted OHLC prices with columns ["beta_adj_open", "beta_adj_high", "beta_adj_low", "beta_adj_close", "residual_log_return", "beta"]
    Raises:
    ValueError: If there are no common dates between the market data and the benchmark data.
    """
    [md, md_benchmark] = align_dataframes_on_common_dates([md, md_benchmark])
    required_keys = ["open", "high", "low", "close"]
    if not all(key in columns for key in required_keys):
        get_ohlcutils_logger().log_error(f"The `columns` argument must include mappings for {required_keys}.", ValueError(f"The `columns` argument must include mappings for {required_keys}."), {
            "required_keys": required_keys,
            "provided_keys": list(columns.keys()),
            "missing_keys": [key for key in required_keys if key not in columns],
            "function": "calculate_beta_adjusted_bars"
        })
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    open_col = columns["open"]
    high_col = columns["high"]
    low_col = columns["low"]
    close_col = columns["close"]

    # Ensure the mapped columns exist in the DataFrame
    if not all(col in md.columns for col in [open_col, high_col, low_col, close_col]):
        get_ohlcutils_logger().log_error("Mapped columns must exist in the DataFrame.", ValueError("Mapped columns must exist in the DataFrame."), {
            "required_columns": [open_col, high_col, low_col, close_col],
            "available_columns": list(md.columns),
            "missing_columns": [col for col in [open_col, high_col, low_col, close_col] if col not in md.columns],
            "function": "calculate_beta_adjusted_bars"
        })
        raise ValueError("Mapped columns must exist in the DataFrame.")

    # Check for zero values in the first open columns to avoid division by zero
    if md.iloc[0][open_col] == 0 or md_benchmark.iloc[0][open_col] == 0:
        get_ohlcutils_logger().log_error("The first open value in md or md_benchmark is zero, cannot calculate ratio bars.", ValueError("The first open value in md or md_benchmark is zero, cannot calculate ratio bars."), {
            "md_first_open": md.iloc[0][open_col],
            "md_benchmark_first_open": md_benchmark.iloc[0][open_col],
            "function": "calculate_beta_adjusted_bars"
        })
        raise ValueError("The first open value in md or md_benchmark is zero, cannot calculate ratio bars.")

    # Compute beta
    md["beta"] = calculate_beta(md, md_benchmark, window=beta_days, columns=columns)

    # Compute log returns
    md["log_return"] = np.log(md[close_col] / md[close_col].shift(1))
    md_benchmark["market_log_return"] = np.log(md_benchmark[close_col] / md_benchmark[close_col].shift(1))

    # Compute residual returns: idiosyncratic return
    md["residual_log_return"] = md["log_return"] - md["beta"] * md_benchmark["market_log_return"]

    # Reconstruct adjusted prices from residual returns
    md["beta_adj_close"] = md[close_col].iloc[0] * np.exp(md["residual_log_return"].cumsum())

    # Adjust OHLC by same ratio as close adjustment
    adjustment_factor = md["beta_adj_close"] / md[close_col]
    md["beta_adj_open"] = md[open_col] * adjustment_factor
    md["beta_adj_high"] = md[high_col] * adjustment_factor
    md["beta_adj_low"] = md[low_col] * adjustment_factor

    out = md[["beta_adj_open", "beta_adj_high", "beta_adj_low", "beta_adj_close", "residual_log_return", "beta"]].copy()
    return out


def get_heikin_ashi(md, len2_ha=10):
    """Returns heikin ashi candlesticks

    Args:
        df (pd.DataFrame): market data
        len2_ha (int, optional): Smoothing Period. Defaults to 10.

    Returns:
        _type_: _description_
    """
    md = create_index_if_missing(md)
    # Calculate Heikin-Ashi values
    ha_df = pd.DataFrame(index=md.index)
    ha_df["close"] = (md["open"] + md["high"] + md["low"] + md["close"]) / 4
    # Calculate ha_df['open'] recursively
    ha_df["open"] = md["open"].copy()  # Start with the original open values
    index_of_open = ha_df.columns.get_loc("open")
    for i in range(1, len(md)):
        ha_df.iloc[i, index_of_open] = (ha_df["open"].iloc[i - 1] + ha_df["close"].iloc[i - 1]) / 2
    ha_df["high"] = np.maximum.reduce([md["high"], ha_df["close"], ha_df["open"]])
    ha_df["low"] = np.minimum.reduce([md["low"], ha_df["close"], ha_df["open"]])

    # Apply EMA to Heikin-Ashi values
    ha_df["o2"] = ha_df["open"].ewm(span=len2_ha, adjust=False).mean()
    ha_df["c2"] = ha_df["close"].ewm(span=len2_ha, adjust=False).mean()
    ha_df["h2"] = ha_df["high"].ewm(span=len2_ha, adjust=False).mean()
    ha_df["l2"] = ha_df["low"].ewm(span=len2_ha, adjust=False).mean()

    return ha_df


def degree_slope(
    md: pd.DataFrame,
    window: int,
    columns: list = ["asettle"],
    prefix: str = "deg",
    method: str = "simple",
) -> pd.DataFrame:
    """
    Calculates the slope for specified columns over a given window using either
    a simple geometric approach or linear regression.

    Args:
        md (pd.DataFrame): Market data DataFrame with a datetime index.
        window (int): Rolling window size for slope calculation.
        columns (list): List of column names to calculate the slope for.
        prefix (str, optional): Prefix for the resulting column names. Defaults to "deg".
        method (str, optional): Method to calculate the slope. Options are "simple" or "regression".
                                Defaults to "simple".

    Returns:
        pd.DataFrame: DataFrame with the same rows as `md` and columns renamed with the given prefix.
    """
    md = create_index_if_missing(md)
    if not all(col in md.columns for col in columns):
        get_ohlcutils_logger().log_error("All specified columns must exist in the DataFrame.", ValueError("All specified columns must exist in the DataFrame."), {
            "required_columns": columns,
            "available_columns": list(md.columns),
            "missing_columns": [col for col in columns if col not in md.columns],
            "function": "degree_slope"
        })
        raise ValueError("All specified columns must exist in the DataFrame.")
    if window < 2:
        get_ohlcutils_logger().log_error("Window size must be at least 2.", ValueError("Window size must be at least 2."), {
            "window": window,
            "function": "degree_slope"
        })
        raise ValueError("Window size must be at least 2.")
    if method not in ["simple", "regression"]:
        get_ohlcutils_logger().log_error("Method must be either 'simple' or 'regression'.", ValueError("Method must be either 'simple' or 'regression'."), {
            "method": method,
            "valid_methods": ["simple", "regression"],
            "function": "degree_slope"
        })
        raise ValueError("Method must be either 'simple' or 'regression'.")

    result = pd.DataFrame(index=md.index)

    for col in columns:
        if method == "simple":
            # Simple geometric slope calculation
            slopes = (
                md[col]
                .rolling(window=window)
                .apply(lambda x: np.degrees(np.arctan((x[-1] - x[0]) / (window - 1))), raw=True)
            )
        elif method == "regression":
            # Linear regression slope calculation
            def calc_regression_slope(y):
                x = np.arange(len(y)).reshape(-1, 1)
                model = LinearRegression().fit(x, y)
                return np.degrees(np.arctan(model.coef_[0]))

            slopes = md[col].rolling(window=window).apply(calc_regression_slope, raw=False)

        result[f"{prefix}_{col}"] = slopes

    return result


def average_band(
    md: pd.DataFrame,
    size: int = 100,
    ema: int = 9,
    columns: dict = {"high": "ahigh", "low": "alow", "close": "asettle"},
) -> pd.DataFrame:
    """
    Returns the average band, Bollinger Bands, highest high, lowest low, and displacement around the average band,
    inspired by the Average Band by Harman in TradingView.

    Args:
        md (pd.DataFrame): DataFrame of market data (OHLC).
        size (int, optional): Window size for calculating bands. Defaults to 100.
        ema (int, optional): EMA period for the shortest line. Defaults to 9.
        columns (dict, optional): Mapping of column names in `md` to the ones used in the function.
                                  Defaults to {"high": "high", "low": "low", "close": "close"}.

    Returns:
        pd.DataFrame: DataFrame with calculated bands, slopes, Bollinger Bands, highest high, lowest low, and displacement.
        Columns: [Upper_Band,Middle_Band,Lower_Band,Upper_Band_Slope,Middle_Band_Slope,Lower_Band_Slope,
                  Avg_Band_Slope,BB_Middle,BB_Upper,BB_Lower,Highest_High,Lowest_Low,Shortest,displacement]
    """
    # Ensure the required columns are present in the DataFrame
    required_keys = ["high", "low", "close"]
    if not all(key in columns for key in required_keys):
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    high_col = columns["high"]
    low_col = columns["low"]
    close_col = columns["close"]

    # Ensure the mapped columns exist in the DataFrame
    if not all(col in md.columns for col in [high_col, low_col, close_col]):
        raise ValueError("Mapped columns must exist in the DataFrame.")

    md = create_index_if_missing(md)
    # Calculating the Simple Moving Averages (SMA)
    md["Upper_Band"] = md[high_col].rolling(window=size).mean()
    md["Middle_Band"] = (md[high_col].rolling(window=size).mean() + md[low_col].rolling(window=size).mean()) / 2
    md["Lower_Band"] = md[low_col].rolling(window=size).mean()

    # Calculate Bollinger Bands
    md["BB_Middle"] = md[close_col].rolling(window=size).mean()
    md["BB_StdDev"] = md[close_col].rolling(window=size).std()
    md["BB_Upper"] = md["BB_Middle"] + (2 * md["BB_StdDev"])
    md["BB_Lower"] = md["BB_Middle"] - (2 * md["BB_StdDev"])

    # Calculate highest high and lowest low for the given window size
    md["Highest_High"] = md[high_col].rolling(window=size).max()
    md["Lowest_Low"] = md[low_col].rolling(window=size).min()

    # Calculate average slopes for each band
    md["Upper_Band_Slope"] = (
        md["Upper_Band"]
        .rolling(window=size)
        .apply(lambda x: np.degrees(np.arctan((x[-1] - x[0]) / (size - 1))), raw=True)
    )
    md["Middle_Band_Slope"] = (
        md["Middle_Band"]
        .rolling(window=size)
        .apply(lambda x: np.degrees(np.arctan((x[-1] - x[0]) / (size - 1))), raw=True)
    )
    md["Lower_Band_Slope"] = (
        md["Lower_Band"]
        .rolling(window=size)
        .apply(lambda x: np.degrees(np.arctan((x[-1] - x[0]) / (size - 1))), raw=True)
    )
    md["Avg_Band_Slope"] = (md["Upper_Band_Slope"] + md["Middle_Band_Slope"] + md["Lower_Band_Slope"]) / 3

    # EMA for the shortest line near candles
    md["Shortest"] = md[close_col].ewm(span=ema, adjust=False).mean()

    # Calculate displacement
    long_displacement = (md["Shortest"] - md["Lower_Band"]) * 100 / md["Shortest"]
    short_displacement = (md["Shortest"] - md["Upper_Band"]) * 100 / md["Shortest"]

    md["displacement"] = np.where(
        (long_displacement > 0) & (short_displacement < 0),
        0,
        np.where(short_displacement < 0, short_displacement, long_displacement),
    )

    return md[
        [
            "Upper_Band",
            "Middle_Band",
            "Lower_Band",
            "Upper_Band_Slope",
            "Middle_Band_Slope",
            "Lower_Band_Slope",
            "Avg_Band_Slope",
            "BB_Middle",
            "BB_Upper",
            "BB_Lower",
            "Highest_High",
            "Lowest_Low",
            "Shortest",
            "displacement",
        ]
    ]


def shift(source: np.array, signal: np.array, ref: int):
    """Helper function for trend function. Used to create a shifted
    source (say swinghh), the last time the signal changed from ref value
    to another value.

    (say signal = updownbar changed to ref = 1, i.e. a new upswing started.
    We want to know the swinghh at the culmination of prior upswing. The function
    returns this prior swinghh)

    Args:
        source (np.array): Source which has information to be shifted
        signal (np.array): The array that holds trigger value for calculating shift
        ref (int): The trigger value

    Returns:
        Array: Shifted Array
    """
    out = np.zeros(len(source))
    for i in range(1, len(source)):
        trigger = (signal[i] != signal[i - 1]) and signal[i] == ref
        if (source[i] != source[i - 1]) or trigger:
            out[i] = source[i - 1]
        else:
            out[i] = out[i - 1]
    return out


def trend(md, bars=1, columns: dict = {"high": "ahigh", "low": "alow"}) -> pd.DataFrame:
    """
    Calculate the trend and swing levels for a given OHLC (Open-High-Low-Close) dataset.
    This function identifies trends, swing highs, swing lows, and other related metrics
    based on the provided OHLC data. It uses a bar-based approach to determine the
    direction of the trend and classify bars as up, down, inside, or outside.
    Parameters:
        md (pd.DataFrame):
            A pandas DataFrame containing the OHLC data. The column names for high, low,
            and close prices should be mapped using the `columns` parameter.
        bars (int, optional):
            The number of bars to consider for calculating the trend. Default is 1.
        columns (dict, optional):
            A dictionary mapping the required column names in the DataFrame.
            Keys should include "high", "low", and "close", and their corresponding
            values should map to the actual column names in the DataFrame.
            Default is {"high": "high", "low": "low", "close": "close"}.
    Returns:
        pd.DataFrame:
            A DataFrame containing the following columns:
            - "trend": The calculated trend direction (1 for uptrend, -1 for downtrend, 0 otherwise).
            - "swingtype": Indicates the type of swing (1 for upswing, -1 for downswing).
            - "daysinswing": The number of days in the current swing.
            - "updownbar": Indicates if the bar is an up bar (1), down bar (-1), or neutral (0).
            - "outsidebar": Indicates if the bar is an outside bar (1 for true, 0 otherwise).
            - "insidebar": Indicates if the bar is an inside bar (1 for true, 0 otherwise).
            - "swinghigh": The swing high value for the current bar.
            - "swinglow": The swing low value for the current bar.
            - "swinghh": The highest swing value in the prior upswing. final set after new downswing starts
            - "swingll": The lowest swing value in the prior downswing. finat set after new upswing starts
            - "swinghh_1": prior swinghh value which waa different
            - "swingll_1": prior swingll value which waa different
    Raises:
        ValueError:
            - If the `columns` dictionary does not include mappings for "high", "low", and "close".
            - If the mapped columns do not exist in the DataFrame.
            - If the length of the high series is less than or equal to the number of bars.
    Notes:
        - The function uses numpy for efficient array operations.
        - The trend calculation is based on a combination of bar patterns and swing levels.
        - The first row of the returned DataFrame may contain default or placeholder values
          due to the nature of the calculations.
    Example:
        >>> data = pd.DataFrame({
        ...     "high": [10, 12, 11, 13],
        ...     "low": [8, 9, 10, 11],
        ...     "close": [9, 11, 10, 12]
        ... })
        >>> trend(data, bars=2)
    """

    # Ensure the required columns are present in the DataFrame
    md = create_index_if_missing(md)
    required_keys = ["high", "low"]
    if not all(key in columns for key in required_keys):
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    high_col = columns["high"]
    low_col = columns["low"]

    # Ensure the mapped columns exist in the DataFrame
    if not all(col in md.columns for col in [high_col, low_col]):
        raise ValueError("Mapped columns must exist in the DataFrame.")

    high = md[high_col]
    low = md[low_col]

    if len(high) <= bars:
        raise ValueError("High timeseries is too short to calculate trend")

    result, updownbarclean, updownbar, outsidebar, insidebar, swingtype, daysinswing, trendtype = np.zeros(
        (8, len(high))
    )
    swinglevel, swinghigh, swinghh, swinghh_1, swinglow, swingll, swingll_1 = np.zeros((7, len(high)))
    swingLowStartIndex, swingHighStartIndex = (0, 0)
    high_shift = high.shift(1)
    low_shift = low.shift(1)

    for i in range(2, bars + 1):
        high_shift = np.maximum(high_shift, high.shift(i))
        low_shift = np.minimum(low_shift, low.shift(i))

    hh = np.where(high > high_shift, 1, 0)
    lh = np.where(high < high_shift, 1, 0)
    ll = np.where(low < low_shift, 1, 0)
    hl = np.where(low > low_shift, 1, 0)
    el = np.where(low == low_shift, 1, 0)
    eh = np.where(high == high_shift, 1, 0)

    updownbar = np.where(hh * hl + eh * hl + hh * el == 1, 1, 0)
    updownbarclean = np.where(hh * hl + eh * hl + hh * el == 1, 1, 0)
    updownbar = np.where(lh * ll + el * lh + ll * eh == 1, -1, updownbar)
    updownbarclean = np.where(lh * ll + el * lh + ll * eh == 1, -1, updownbarclean)
    outsidebar = np.where(hh * ll == 1, 1, 0)
    insidebar = np.where((updownbar == 0) & (outsidebar == 0), 1, 0)

    for i in range(1, len(high)):
        if outsidebar[i] == 1:
            updownbar[i] = -updownbar[i - 1]
        elif insidebar[i] == 1:
            updownbar[i] = updownbar[i - 1]
        else:
            pass

    for i in range(1, len(high)):
        if updownbar[i] == 1:
            swingtype[i] = 1
            if updownbar[i - 1] == 1:  # continued upswing
                daysinswing[i] = daysinswing[i - 1] + 1
                swinghigh[i] = high.iloc[i] if high.iloc[i] > swinghigh[i - 1] else swinghigh[i - 1]
                if swinglow[i - 1] != 0 and outsidebar[i] != 1:
                    swinglow[i] = swinglow[i - 1]
                    swingll[i] = swinglow[i - 1]
            else:  # first day of upswing
                daysinswing[i] = 1
                swingHighStartIndex = i
                swinghigh[i] = high.iloc[i]
                # start of new swing. Free prior swing levels
                if outsidebar[i] == 1:
                    swinglow[i] = min(low.iloc[i], swinglow[i - 1])
                    swingll[i] = swinglow[i]
                else:
                    swinglow[i] = swinglow[i - 1]
                    swingll[i] = swinglow[i - 1]
                for j in range(swingLowStartIndex, swingHighStartIndex):
                    # set swingll for the downswing that has just completed.
                    swingll[j] = swinglow[i]
        else:  # updownbar[i] == -1 :
            swingtype[i] = -1
            if updownbar[i - 1] == -1:  # continued downswing
                daysinswing[i] = daysinswing[i - 1] + 1
                swinglow[i] = min(low.iloc[i], swinglow[i - 1])
                if swinghigh[i - 1] != 0 and outsidebar[i] != 1:
                    swinghigh[i] = swinghigh[i - 1]
                    swinghh[i] = swinghigh[i - 1]
            else:  # first day of dnswing
                daysinswing[i] = 1
                swingLowStartIndex = i
                swinglow[i] = low.iloc[i]
                if outsidebar[i] == 1:
                    swinghigh[i] = max(high.iloc[i], swinghigh[i - 1])
                    swinghh[i] = swinghigh[i]
                else:
                    swinghigh[i] = swinghigh[i - 1]
                    swinghh[i] = swinghigh[i - 1]
                for j in range(swingHighStartIndex, swingLowStartIndex):
                    swinghh[j] = swinghigh[i]

    # update swinghh, swingll for last (incomplete) swing
    if swingHighStartIndex > swingLowStartIndex:  # last incomplete swing is up
        for j in range(swingHighStartIndex, len(high)):
            swinghh[j] = swinghigh[len(high) - 1]
    else:
        for j in range(swingLowStartIndex, len(high)):
            swingll[j] = swinglow[len(high) - 1]

    # calculate shifted versions of highhigh and lowlow
    swingll_1 = shift(swingll, updownbar, -1)
    swinghh_1 = shift(swinghh, updownbar, 1)
    swingll_2 = shift(swingll_1, updownbar, -1)
    swinghh_2 = shift(swinghh_1, updownbar, 1)

    # create swing level
    for i in range(1, len(high)):
        if updownbar[i] == 1:
            swinglevel[i] = swinghh[i]
        elif updownbar[i] == -1:
            swinglevel[i] = swingll[i]
        else:
            swinglevel[i] = 0
    # ensure first swing level is never zero else it goes outside plot
    swinglevel[0] = swinglevel[1]

    # update trend
    for i in range(1, len(high)):
        up1 = (updownbar[i] == 1) and swinghigh[i] > swinghh_1[i] and swinglow[i] > swingll_1[i]
        up2 = (updownbar[i] == 1) and swinghh_1[i] > swinghh_2[i] and swinglow[i] > swingll_1[i]
        up3 = (
            (updownbar[i] == -1 or outsidebar[i] == 1)
            and swinghigh[i] > swinghh_1[i]
            and swingll_1[i] > swingll_2[i]
            and swinglow[i] > swingll_1[i]
            and low.iloc[i] > swingll_1[i]
        )
        down1 = (updownbar[i] == -1) and swinghigh[i] < swinghh_1[i] and swinglow[i] < swingll_1[i]
        down2 = (updownbar[i] == -1) and swinghigh[i] < swinghh_1[i] and swingll_1[i] < swingll_2[i]
        down3 = (
            (updownbar[i] == 1 or outsidebar[i] == 1)
            and swinghh_1[i] < swinghh_2[i]
            and swinglow[i] < swingll_1[i]
            and swinghigh[i] < swinghh_1[i]
            and high.iloc[i] < swinghh_1[i]
        )

        if up1 or up2 or up3:
            result[i] = 1
            if up1:
                trendtype[i] = 1
            elif up2:
                trendtype[i] = 2
            else:
                trendtype[i] = 3
        elif down1 or down2 or down3:
            result[i] = -1
            if down1:
                trendtype[i] = 1
            elif down2:
                trendtype[i] = 2
            else:
                trendtype[i] = 3

    return pd.DataFrame(
        {
            "trend": result,
            "swingtype": swingtype,
            "daysinswing": daysinswing,
            "updownbar": updownbar,
            "outsidebar": outsidebar,
            "insidebar": insidebar,
            "swinghigh": swinghigh,
            "swinglow": swinglow,
            "swinghh": swinghh,
            "swingll": swingll,
            "swinghh_1": swinghh_1,
            "swingll_1": swingll_1,
        },
        index=high.index,
    )


def range_filter(
    md,
    per: int = 100,
    mult: int = 3,
    columns: dict = {"close": "asettle"},
) -> pd.DataFrame:
    """
    Applies a range filter to the given market data (md) DataFrame.
    The range filter calculates a smoothed range and determines the midrange,
    high range, low range, and directional trends (upward and downward) based
    on the provided parameters.
    Parameters:
        md (pd.DataFrame): The input market data DataFrame. Must contain the
            columns specified in the `columns` argument.
        per (int, optional): The period for calculating the exponential moving
            average (EMA). Default is 100.
        mult (int, optional): The multiplier for the smoothed range. Default is 3.
        columns (dict, optional): A dictionary mapping required keys to column
            names in the input DataFrame. The default is {"close": "asettle"}.
            Required keys:
                - "close": The column representing the closing prices.
    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - "smoothrng": The smoothed range values.
            - "midrange": The midrange values (range filter).
            - "highrange": The upper boundary of the range.
            - "lowrange": The lower boundary of the range.
            - "upward": The upward trend count.
            - "downard": The downward trend count.
    Raises:
        ValueError: If the `columns` argument does not include mappings for
            the required keys or if the mapped columns do not exist in the
            input DataFrame.
    Notes:
        - The function uses the `ta.ema` method for calculating the exponential
          moving average. Ensure the required library is imported and available.
        - The input DataFrame is expected to have a proper index. If missing,
          an index will be created using the `create_index_if_missing` function.
    """

    required_keys = ["close"]
    if not all(key in columns for key in required_keys):
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    close_col = columns["close"]

    # Ensure the mapped columns exist in the DataFrame
    if not all(col in md.columns for col in [close_col]):
        raise ValueError("Mapped columns must exist in the DataFrame.")

    md = create_index_if_missing(md)
    close = md.loc[:, close_col]

    wper = 2 * per - 1
    avgrng = ta.ema(abs(close - close.shift(1).bfill()), per).bfill()
    smoothrng = ta.ema(avgrng, timeperiod=wper).fillna(0) * mult
    rngfilt = close.copy()  # Create a copy to avoid modifying a slice
    for i in range(2, len(rngfilt)):
        if close.iloc[i] > rngfilt.iloc[i - 1]:  # upward market
            if (close.iloc[i] - smoothrng.iloc[i]) < rngfilt.iloc[
                i - 1
            ]:  # close is within the smoothing range of rangefilter.
                rngfilt.iloc[i] = rngfilt.iloc[i - 1]  # dont adjust rangefilter
            else:
                rngfilt.iloc[i] = close.iloc[i] - smoothrng.iloc[i]  # set rangefilter to new, higher value
        else:
            if (close.iloc[i] + smoothrng.iloc[i]) > rngfilt.iloc[i - 1]:
                rngfilt.iloc[i] = rngfilt.iloc[i - 1]
            else:
                rngfilt.iloc[i] = close.iloc[i] + smoothrng.iloc[i]
    upward = np.zeros(len(rngfilt))
    downward = np.zeros(len(rngfilt))
    for i in range(2, len(rngfilt)):
        if rngfilt.iloc[i] > rngfilt.iloc[i - 1]:
            upward[i] = upward[i - 1] + 1
        elif rngfilt.iloc[i] < rngfilt.iloc[i - 1]:
            upward[i] = 0
        else:
            upward[i] = upward[i - 1]

    if rngfilt.iloc[i] < rngfilt.iloc[i - 1]:
        downward[i] = downward[i - 1] + 1
    elif rngfilt.iloc[i] > rngfilt.iloc[i - 1]:
        downward[i] = 0
    else:
        downward[i] = downward[i - 1]
    out = pd.DataFrame(
        {
            "smoothrng": smoothrng,
            "midrange": rngfilt,
            "highrange": rngfilt + smoothrng,
            "lowrange": rngfilt - smoothrng,
            "upward": upward,
            "downward": downward,
        },
        index=md.index,
    )
    return out


def t3ma(
    md: pd.DataFrame,
    len: int = 5,
    volume_factor: float = 0.7,
    columns: dict = {"close": "asettle"},
) -> pd.DataFrame:
    """
    Calculates the Tilson T3 moving average for the specified column(s) in the DataFrame.

    Args:
        md (pd.DataFrame): Market data DataFrame.
        len (int, optional): Period for the Tilson T3 Moving Average. Defaults to 5.
        volume_factor (float, optional): Volume Factor for the Tilson T3 moving average. Normally 0.7 or 0.618. Defaults to 0.7.
        columns (dict, optional): Mapping of column names in `md` to the ones used in the function.
                                  Defaults to {"close": "close"}.

    Returns:
        pd.DataFrame: DataFrame with the calculated T3 moving average for the specified column(s).
    """
    # Ensure the required columns are present in the DataFrame
    required_keys = ["close"]
    if not all(key in columns for key in required_keys):
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    close_col = columns["close"]

    # Ensure the mapped columns exist in the DataFrame
    if close_col not in md.columns:
        raise ValueError(f"The column '{close_col}' specified in `columns` does not exist in the DataFrame.")

    # Ensure the DataFrame has a datetime index
    md = create_index_if_missing(md)

    # Extract the source column
    src = md[close_col]

    # Calculate the T3 moving average
    xe1 = ta.ema(src, len)
    xe2 = ta.ema(xe1, len)
    xe3 = ta.ema(xe2, len)
    xe4 = ta.ema(xe3, len)
    xe5 = ta.ema(xe4, len)
    xe6 = ta.ema(xe5, len)

    b = volume_factor
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b

    t3 = c1 * xe6 + c2 * xe5 + c3 * xe4 + c4 * xe3

    # Return the result as a DataFrame
    return pd.DataFrame({f"t3_{close_col}": t3}, index=md.index)


def bextrender(
    md: pd.DataFrame,
    short_period: int = 5,
    long_period: int = 20,
    rsi_period: int = 15,
    t3_ma_len: int = 5,
    t3_ma_volume_factor: float = 0.7,
    columns: dict = {"close": "asettle"},
) -> pd.DataFrame:
    """
    Calculates B-Xtrender values for the given market data.

    Args:
        md (pd.DataFrame): Market data DataFrame.
        short_period (int, optional): Short period for EMA. Defaults to 5.
        long_period (int, optional): Long period for EMA. Defaults to 20.
        rsi_period (int, optional): Period for rsi. Defaults to 15.
        t3_ma_len (int, optional): Period for Tilson T3 Moving Average. Defaults to 5.
        t3_ma_volume_factor (float, optional): Volume Factor for Tilson T3 moving average. Normally 0.7 or 0.618. Defaults to 0.7.
        columns (dict, optional): Mapping of column names in `md` to the ones used in the function.
                                  Defaults to {"close": "close"}.

    Returns:
        pd.DataFrame: DataFrame with the calculated B-Xtrender values:
                      - "shortTermXtrender"
                      - "shortT3MA"
                      - "longTermXtrender"
                      - "longT3MA"
    """
    # Ensure the required columns are present in the DataFrame
    required_keys = ["close"]
    if not all(key in columns for key in required_keys):
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    close_col = columns["close"]

    # Ensure the mapped columns exist in the DataFrame
    if close_col not in md.columns:
        raise ValueError(f"The column '{close_col}' specified in `columns` does not exist in the DataFrame.")

    # Ensure the DataFrame has a datetime index
    md = create_index_if_missing(md)

    # Extract the source column
    src = md[close_col]

    # Check if the length of the source data is sufficient
    if len(src) <= long_period + rsi_period - 1:
        nan_values = pd.Series(np.nan, index=md.index)
        return pd.DataFrame(
            {
                "shortTermXtrender": nan_values,
                "shortT3MA": nan_values,
                "longTermXtrender": nan_values,
                "longT3MA": nan_values,
            }
        )

    # Calculate short-term and long-term Xtrender
    shortTermXtrender = ta.rsi(ta.ema(src, short_period) - ta.ema(src, long_period), rsi_period) - 50
    longTermXtrender = ta.rsi(ta.ema(src, long_period), rsi_period) - 50

    # Calculate T3 moving averages for short-term and long-term Xtrender
    shortT3MA = t3ma(
        pd.DataFrame({close_col: shortTermXtrender}),
        len=t3_ma_len,
        volume_factor=t3_ma_volume_factor,
        columns={"close": close_col},
    )[f"t3_{close_col}"]

    longT3MA = t3ma(
        pd.DataFrame({close_col: longTermXtrender}),
        len=t3_ma_len,
        volume_factor=t3_ma_volume_factor,
        columns={"close": close_col},
    )[f"t3_{close_col}"]

    # Return the result as a DataFrame
    return pd.DataFrame(
        {
            "shortTermXtrender": shortTermXtrender,
            "shortT3MA": shortT3MA,
            "longTermXtrender": longTermXtrender,
            "longT3MA": longT3MA,
        },
        index=md.index,
    )


def vwap(
    md: pd.DataFrame,
    periods: int = 21,
    columns: dict = {"close": "asettle", "volume": "avolume"},
) -> pd.DataFrame:
    """
    Calculates the Volume Weighted Average Price (VWAP) for the given market data.

    Args:
        md (pd.DataFrame): Market data DataFrame.
        periods (int, optional): Rolling window size for VWAP calculation. Defaults to 21.
        columns (dict, optional): Mapping of column names in `md` to the ones used in the function.
                                  Defaults to {"price": "close", "volume": "volume"}.

    Returns:
        pd.DataFrame: DataFrame with the calculated VWAP as a new column.
    """
    # Ensure the required columns are present in the DataFrame
    required_keys = ["close", "volume"]
    if not all(key in columns for key in required_keys):
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    close_col = columns["close"]
    volume_col = columns["volume"]

    # Ensure the mapped columns exist in the DataFrame
    if not all(col in md.columns for col in [close_col, volume_col]):
        raise ValueError("Mapped columns must exist in the DataFrame.")

    # Ensure the DataFrame has a datetime index
    md = create_index_if_missing(md)

    # Extract the source columns
    src_price = md[close_col]
    src_volume = md[volume_col]

    # Calculate VWAP
    traded_value = src_price * src_volume
    vwap_data = (
        traded_value.rolling(window=periods, min_periods=1).sum()
        / src_volume.rolling(window=periods, min_periods=1).sum()
    )

    # Return the result as a DataFrame
    return pd.DataFrame({f"vwap_{close_col}": vwap_data}, index=md.index)


def calc_rolling(x: pd.Series, periods: int, indicator: Callable, column_name: str = "rolling") -> pd.DataFrame:
    """
    Applies a custom indicator function to a rolling window of a Pandas Series and returns a DataFrame.

    Args:
        x (pd.Series): The input Pandas Series on which the rolling calculation will be applied.
        periods (int): The size of the rolling window.
        indicator (callable): A function or callable object that computes the desired metric or indicator for each rolling window.
        column_name (str, optional): The name of the resulting column in the returned DataFrame. Defaults to "rolling".

    Returns:
        pd.DataFrame: A DataFrame with the same index as `x` and a single column containing the calculated rolling values.

    Example 1:
    # Define a simple indicator function (mean)
    rolling_mean_df = calc_rolling(df["value"], periods=3, indicator=lambda x: x.mean(), column_name="rolling_mean")


    # Define a custom indicator function for RSI using a pandas ta indicator
    rolling_rsi_df = calc_rolling(
        df["close"],
        periods=5,
        indicator=lambda x: ta.rsi(pd.Series(x), length=5).iloc[-1],  # Use pandas_ta RSI
        column_name="rolling_rsi"
    )
    """
    # Apply the rolling calculation
    rolling_values = x.rolling(periods, min_periods=2).apply(indicator)

    # Return as a DataFrame
    return pd.DataFrame({column_name: rolling_values}, index=x.index)


def hilega_milega(
    md: pd.DataFrame,
    rsi_days: int = 9,
    ma_days: int = 21,
    ema_days: int = 3,
    columns: dict = {"close": "asettle"},
) -> pd.DataFrame:
    """
    Calculates the Hilega-Milega indicator based on RSI, EMA, and WMA.

    Args:
        md (pd.DataFrame): Market data DataFrame.
        rsi_days (int, optional): Period for RSI calculation. Defaults to 9.
        ma_days (int, optional): Period for WMA calculation. Defaults to 21.
        ema_days (int, optional): Period for EMA calculation. Defaults to 3.
        columns (dict, optional): Mapping of column names in `md` to the ones used in the function.
                                  Defaults to {"close": "asettle"}.

    Returns:
        pd.DataFrame: DataFrame with the calculated Hilega-Milega indicator and its components:
                      - "hilega_milega_ind": The Hilega-Milega indicator (1 or -1).
                      - "rsi": The RSI values.
                      - "ema": The EMA of RSI values.
                      - "wma": The WMA of RSI values.
    """
    # Ensure the required columns are present in the DataFrame
    required_keys = ["close"]
    if not all(key in columns for key in required_keys):
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    close_col = columns["close"]

    # Ensure the mapped columns exist in the DataFrame
    if close_col not in md.columns:
        raise ValueError(f"The column '{close_col}' specified in `columns` does not exist in the DataFrame.")

    # Ensure the DataFrame has a datetime index
    md = create_index_if_missing(md)

    # Extract the source column
    src_price = md[close_col]

    # Calculate RSI, EMA, and WMA
    rsi = ta.rsi(src_price, length=rsi_days)
    ema = ta.ema(rsi, length=ema_days)
    wma = ta.wma(rsi, length=ma_days)

    # Calculate the Hilega-Milega indicator
    hilega_milega_ind = pd.Series(np.where(ema > wma, 1, -1), index=src_price.index, name="hilega_milega_ind")

    # Return the result as a DataFrame
    return pd.DataFrame(
        {
            "hilega_milega_ind": hilega_milega_ind,
            "rsi": rsi,
            "ema": ema,
            "wma": wma,
        },
        index=md.index,
    )


def supertrend(
    md: pd.DataFrame,
    atr_period: int = 14,
    multiplier: float = 3.0,
    columns: dict = {"high": "ahigh", "low": "alow", "close": "asettle"},
) -> pd.DataFrame:
    """
    Calculates the Supertrend indicator for the given market data.

    Args:
        md (pd.DataFrame): Market data DataFrame.
        atr_period (int, optional): Period for ATR calculation. Defaults to 14.
        multiplier (float, optional): Multiplier for ATR to calculate bands. Defaults to 3.0.
        columns (dict, optional): Mapping of column names in `md` to the ones used in the function.
                                  Defaults to {"high": "high", "low": "low", "close": "close"}.

    Returns:
        pd.DataFrame: DataFrame with the calculated Supertrend and its components:
                      - "supertrend": The Supertrend indicator (True for uptrend, False for downtrend).
                      - "final_lowerband": The final lower band.
                      - "final_upperband": The final upper band.
    """
    # Ensure the required columns are present in the DataFrame
    required_keys = ["high", "low", "close"]
    if not all(key in columns for key in required_keys):
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    high_col = columns["high"]
    low_col = columns["low"]
    close_col = columns["close"]

    # Ensure the mapped columns exist in the DataFrame
    if not all(col in md.columns for col in [high_col, low_col, close_col]):
        raise ValueError("Mapped columns must exist in the DataFrame.")

    # Ensure the DataFrame has a datetime index
    md = create_index_if_missing(md)

    # Extract the source columns
    high = md[high_col]
    low = md[low_col]
    close = md[close_col]

    # Calculate ATR
    price_diffs = [high - low, high - close.shift(), close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1).abs().max(axis=1)
    atr = true_range.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()

    # Calculate HL2 (average of high and low prices)
    hl2 = (high + low) / 2

    # Calculate upper and lower bands
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    # Initialize final bands and Supertrend
    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()
    supertrend = [True] * len(md)

    # Calculate Supertrend
    for i in range(1, len(md)):
        if close.iloc[i] > final_upperband.iloc[i - 1]:
            supertrend[i] = True
        elif close.iloc[i] < final_lowerband.iloc[i - 1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i - 1]
            if supertrend[i] and final_lowerband.iloc[i] < final_lowerband.iloc[i - 1]:
                final_lowerband.iloc[i] = final_lowerband.iloc[i - 1]
            if not supertrend[i] and final_upperband.iloc[i] > final_upperband.iloc[i - 1]:
                final_upperband.iloc[i] = final_upperband.iloc[i - 1]

        # Remove bands based on trend direction
        if supertrend[i]:
            final_upperband.iloc[i] = np.nan
        else:
            final_lowerband.iloc[i] = np.nan

    # Return the result as a DataFrame
    return pd.DataFrame(
        {
            "supertrend": supertrend,
            "final_lowerband": final_lowerband,
            "final_upperband": final_upperband,
        },
        index=md.index,
    )


def calc_sr(
    md: pd.DataFrame,
    columns: dict = {"high": "ahigh", "low": "alow"},
) -> pd.DataFrame:
    """
    Calculate Support and Resistance from local turning points.

    Args:
        md (pd.DataFrame): Market data DataFrame.
        columns (dict, optional): Mapping of column names in `md` to the ones used in the function.
                                  Defaults to {"high": "ahigh", "low": "alow"}.

    Returns:
        pd.DataFrame: DataFrame with the following columns:
                      - "support": Support levels (NaN if no support on that row).
                      - "resistance": Resistance levels (NaN if no resistance on that row).
                      - "support_tests": Number of times the support level has been tested.
                      - "resistance_tests": Number of times the resistance level has been tested.
    """
    # Ensure the required columns are present in the DataFrame
    required_keys = ["high", "low"]
    if not all(key in columns for key in required_keys):
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    high_col = columns["high"]
    low_col = columns["low"]

    # Ensure the mapped columns exist in the DataFrame
    if not all(col in md.columns for col in [high_col, low_col]):
        raise ValueError("Mapped columns must exist in the DataFrame.")

    # Ensure the DataFrame has a datetime index
    md = create_index_if_missing(md)

    # Extract the source columns
    high = md[high_col]
    low = md[low_col]

    # Helper functions
    def is_support(i):
        return (
            low.iloc[i] < low.iloc[i - 1]
            and low.iloc[i] < low.iloc[i + 1]
            and low.iloc[i + 1] < low.iloc[i + 2]
            and low.iloc[i - 1] < low.iloc[i - 2]
        )

    def is_resistance(i):
        return (
            high.iloc[i] > high.iloc[i - 1]
            and high.iloc[i] > high.iloc[i + 1]
            and high.iloc[i + 1] > high.iloc[i + 2]
            and high.iloc[i - 1] > high.iloc[i - 2]
        )

    def is_far_from_level(level):
        return all(abs(level - x["level"]) >= s for x in levels)

    # Calculate the average range to determine proximity threshold
    s = np.mean(high - low)

    # Initialize levels and results
    levels = []
    support = [float("nan")] * len(md)
    resistance = [float("nan")] * len(md)
    support_tests = [0] * len(md)
    resistance_tests = [0] * len(md)

    # Identify support and resistance levels
    for i in range(2, len(md) - 2):
        if is_support(i):
            level = low.iloc[i]
            if is_far_from_level(level):
                levels.append({"timestamp": md.index[i], "level": level, "type": "support", "tests": 0})
                support[i] = level
            else:
                for lvl in levels:
                    if abs(level - lvl["level"]) < s and lvl["type"] == "support":
                        lvl["tests"] += 1
                        support_tests[i] = lvl["tests"]
                        support[i] = lvl["level"]
                        break

        if is_resistance(i):
            level = high.iloc[i]
            if is_far_from_level(level):
                levels.append({"timestamp": md.index[i], "level": level, "type": "resistance", "tests": 0})
                resistance[i] = level
            else:
                for lvl in levels:
                    if abs(level - lvl["level"]) < s and lvl["type"] == "resistance":
                        lvl["tests"] += 1
                        resistance_tests[i] = lvl["tests"]
                        resistance[i] = lvl["level"]
                        break

    # Return the result as a DataFrame
    return pd.DataFrame(
        {
            "support": support,
            "resistance": resistance,
            "support_tests": support_tests,
            "resistance_tests": resistance_tests,
        },
        index=md.index,
    )


def srt(
    md: pd.DataFrame,
    days: int = 124,
    columns: dict = {"close": "asettle"},
) -> pd.DataFrame:
    """
    Calculates the SRT (Strength Ratio Trend) indicator, which is the ratio of the current price
    to a specified moving average.

    Args:
        md (pd.DataFrame): Market data DataFrame.
        days (int, optional): Days for the moving average. Defaults to 124.
        columns (dict, optional): Mapping of column names in `md` to the ones used in the function.
                                  Defaults to {"close": "asettle"}.

    Returns:
        pd.DataFrame: DataFrame with the calculated SRT indicator as a new column.
    """
    # Ensure the required columns are present in the DataFrame
    required_keys = ["close"]
    if not all(key in columns for key in required_keys):
        raise ValueError(f"The `columns` argument must include mappings for {required_keys}.")

    # Map the columns
    close_col = columns["close"]

    # Ensure the mapped columns exist in the DataFrame
    if close_col not in md.columns:
        raise ValueError(f"The column '{close_col}' specified in `columns` does not exist in the DataFrame.")

    # Ensure the DataFrame has a datetime index
    md = create_index_if_missing(md)

    # Extract the source column
    src_price = md[close_col]

    # Calculate the SRT indicator
    sma = ta.sma(src_price, length=days)
    srt_indicator = src_price / sma

    # Return the result as a DataFrame
    return pd.DataFrame({f"srt_{close_col}": srt_indicator}, index=md.index)


# ohlcutils

`ohlcutils` is a Python library designed for financial data analysis, focusing on OHLC (Open-High-Low-Close) data. It provides a comprehensive set of tools for calculating indicators, resampling data, and performing advanced market data transformations.

---

## Features

- **Indicators**: A wide range of technical indicators, including moving averages, beta calculations, and trend analysis.
- **Data Resampling**: Flexible utilities for changing timeframes and aligning data.
- **Support and Resistance**: Tools for identifying key levels in market data.
- **Beta and Ratio Adjustments**: Functions for adjusting OHLC data based on benchmarks or beta values.
- **Supertrend and VWAP**: Built-in implementations of popular trading indicators.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repo/ohlcutils.git
cd ohlcutils
pip install -r requirements.txt
```

---

## Modules Overview

Below is a list of available functions in each module. For detailed usage, use the `help(function_name)` command in Python.

### `indicators` Module
- `align_dataframes_on_common_dates(dataframes)`
- `calculate_beta(md, md_benchmark, col="close", window=252)`
- `get_heikin_ashi(md, len2_ha=10)`
- `degree_slope(md, window, columns, prefix="deg", method="simple")`
- `average_band(md, size=100, ema=9, columns={"high": "high", "low": "low", "close": "close"})`
- `trend(md, bars=1, columns={"high": "high", "low": "low"})`
- `range_filter(md, per=100, mult=3, columns={"close": "asettle"})`
- `t3ma(md, len=5, volume_factor=0.7, columns={"close": "close"})`
- `bextrender(md, short_period=5, long_period=20, rsi_period=15, t3_ma_len=5, t3_ma_volume_factor=0.7, columns={"close": "close"})`
- `vwap(md, periods=21, columns={"price": "close", "volume": "volume"})`
- `calc_rolling(x, periods, indicator, column_name="rolling")`
- `hilega_milega(md, rsi_days=9, ma_days=21, ema_days=3, columns={"price": "close"})`
- `supertrend(md, atr_period=14, multiplier=3.0, columns={"high": "high", "low": "low", "close": "close"})`
- `calc_sr(md, columns={"high": "ahigh", "low": "alow"})`
- `srt(md, days=124, columns={"price": "close"})`

### `data` Module
- `get_linked_symbols(short_symbol, complete=False)`
- `get_split_info(short_symbol)`
- `load_symbol(symbol, **kwargs)`
- `change_timeframe(md, dest_bar_size, bar_start_time_in_min="15min", exchange="NSE", label="left", fill="ffill", ...)`
- `calculate_ratio_bars(md, md_benchmark, open_col="aopen", high_col="ahigh", low_col="alow", close_col="asettle")`
- `calculate_beta_adjusted_bars(md, md_benchmark, beta_calc_days=252, open_col="aopen", high_col="ahigh", low_col="alow", close_col="asettle")`

---

## Example Usage

### Calculate Beta
```python
from ohlcutils.indicators import calculate_beta
import pandas as pd

# Load market data
md = pd.read_csv("market_data.csv", parse_dates=["date"], index_col="date")
md_benchmark = pd.read_csv("benchmark_data.csv", parse_dates=["date"], index_col="date")

# Calculate rolling beta
beta = calculate_beta(md, md_benchmark, col="close", window=252)
print(beta)
```

### Resample Data
```python
from ohlcutils.data import change_timeframe

# Resample market data to 1-hour bars
resampled_data = change_timeframe(md, dest_bar_size="1H", exchange="NSE", label="left", fill="ffill")
print(resampled_data)
```

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

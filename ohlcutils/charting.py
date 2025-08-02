from typing import Dict

import pandas as pd
import pandas_ta as ta
import plotly.colors
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .config import get_config
from .indicators import create_index_if_missing

# Import ohlcutils_logger lazily to avoid circular import
def get_ohlcutils_logger():
    """Get ohlcutils_logger instance to avoid circular imports."""
    from . import ohlcutils_logger
    return ohlcutils_logger


def get_dynamic_config():
    return get_config()


pio.renderers.default = get_dynamic_config().get("chart_rendering")


def plot(
    df_list,
    candle_sticks=None,  # Dict[int, {"df_idx": int, "candle_stick_columns": {...}, "yaxis": "y"}]
    df_features=None,  # Dict: {pane_num: [ { "df_idx": int, "column": str, "yaxis": str }, ... ] }
    ta_indicators=None,  # Dict: {pane_number: [indicator_dicts]}
    title="",
    max_x_labels=10,
    pane_titles=None,
    pane_heights=None,
) -> go.Figure:
    """
    Multi-pane candlestick and indicator plot using Plotly, supporting up to 2 y-axes per pane.

    Parameters:
    -----------
    df_list : list of pandas.DataFrame
        List of DataFrames containing data to plot. Each DataFrame should have a datetime index.

    candle_sticks : dict, optional
        A dictionary where keys are pane numbers (int) and values are lists of candlestick configurations.
        Each candlestick configuration is a dictionary with the following keys:
        - "df_idx" (int): Index of the DataFrame in `df_list` to use for this candlestick (1-based index).
        - "candle_stick_columns" (dict): A dictionary mapping candlestick components to column names:
            - "open" (str): Column name for open prices.
            - "high" (str): Column name for high prices.
            - "low" (str): Column name for low prices.
            - "close" (str): Column name for close prices.
        - "yaxis" (str, optional): Y-axis to use for this candlestick. Default is "y" (primary y-axis).

    df_features : dict, optional
        A dictionary where keys are pane numbers (int) and values are lists of feature configurations.
        Each feature configuration is a dictionary with the following keys:
        - "df_idx" (int, optional): Index of the DataFrame in `df_list` to use for this feature (1-based index).
          Default is 1.
        - "column" (str): Column name of the feature to plot.
        - "yaxis" (str, optional): Y-axis to use for this feature. Default is "y" (primary y-axis).

    ta_indicators : dict, optional
        A dictionary where keys are pane numbers (int) and values are lists of technical indicator configurations.
        Each indicator configuration is a dictionary with the following keys:
        - "name" (str): Name of the technical indicator (e.g., "sma", "rsi", "atr").
        - "df_idx" (int): Index of the DataFrame in `df_list` to use for this indicator (1-based index).
        - "kwargs" (dict): Keyword arguments to pass to the indicator function (e.g., {"length": 14, "close": "close"}).
        - "column_name" (str, optional): Name to use for the resulting column. Default is the indicator name.
        - "yaxis" (str, optional): Y-axis to use for this indicator. Default is "y" (primary y-axis).
        - "columns" (list of str, optional): List of specific columns to plot if the indicator returns multiple columns.

    title : str, optional
        Title of the chart. Default is an empty string.

    max_x_labels : int, optional
        Maximum number of x-axis labels to display. Default is 10.

    pane_titles : dict, optional
        A dictionary where keys are pane numbers (int) and values are titles (str) for each pane.
        Default is None, which assigns generic titles like "Pane 1", "Pane 2", etc.

    max_yaxes_per_pane : int, optional
        Maximum number of y-axes allowed per pane. Default is 4.

    pane_heights : list of float, optional
        List of relative heights for each pane. The sum of all heights should equal 1.0.
        Default is None, which assigns equal heights to all panes.

    Returns:
    --------
    go.Figure object
    Displays the Plotly chart in the configured renderer.

    Notes:
    ------
    - The `df_idx` parameter in `candle_sticks`, `df_features`, and `ta_indicators` is 1-based, meaning the first
      DataFrame in `df_list` is referenced as `df_idx=1`.
    - The `yaxis` parameter can be "y" (primary y-axis) or "y2", "y3", etc., for secondary y-axes.
    - If `pane_heights` is provided, its length must match the number of panes.
    - The function automatically aligns the indices of all DataFrames in `df_list` to ensure consistent x-axis values.
    """
    for i in range(len(df_list)):
        df_list[i] = create_index_if_missing(df_list[i])

    # determine total panes needed from candle_sticks, df_features, ta_indicators
    pane_keys = []
    if candle_sticks:
        pane_keys.append(max(candle_sticks.keys()))
    if df_features:
        pane_keys.append(max(df_features.keys()))
    if ta_indicators:
        pane_keys.append(max(ta_indicators.keys()))
    n_panes = max(pane_keys) if pane_keys else 1

    # backward‐compat: wrap old single‐pane arg into candle_sticks
    if candle_sticks is None:
        candle_sticks = {
            1: {
                "df_idx": 1,
                "candle_stick_columns": {
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                },
                "yaxis": "y",
            }
        }

    # Calculate row heights
    if pane_heights:
        if len(pane_heights) != n_panes:
            get_ohlcutils_logger().log_error(f"pane_heights must have {n_panes} values, one for each pane.", ValueError(f"pane_heights must have {n_panes} values, one for each pane."), {
                "pane_heights_length": len(pane_heights),
                "n_panes": n_panes,
                "function": "plot"
            })
            raise ValueError(f"pane_heights must have {n_panes} values, one for each pane.")
        row_heights = pane_heights
    else:
        row_heights = [0.5] + [(0.5 / (n_panes - 1))] * (n_panes - 1) if n_panes > 1 else [1.0]

    # Create specs with secondary_y enabled for all panes
    specs = [[{"secondary_y": True}] for _ in range(n_panes)]

    fig = make_subplots(
        rows=n_panes,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        specs=specs,  # Enable secondary y-axes for all panes
        subplot_titles=[
            pane_titles.get(i, f"Pane {i}") if pane_titles else ("Main" if i == 1 else f"Pane {i}")
            for i in range(1, n_panes + 1)
        ],
    )

    # --- Track y-axes per pane ---
    yaxes_dict: Dict[int, Dict[str, dict]] = {pane: {} for pane in range(1, n_panes + 1)}
    yaxis_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Track min/max values for each axis to set proper ranges
    axis_ranges = {
        pane: {"primary": [float("inf"), float("-inf")], "secondary": [float("inf"), float("-inf")]}
        for pane in range(1, n_panes + 1)
    }

    # 1. Candlesticks per pane
    for pane_num, csticks in candle_sticks.items():
        for cs in csticks:
            df = df_list[cs["df_idx"] - 1]
            cols = cs["candle_stick_columns"]
            # decide primary vs. secondary based on yaxis key
            logical_yaxis = cs.get("yaxis", "y")
            axis_idx = int(logical_yaxis[1:]) if logical_yaxis != "y" else 1
            use_secondary_y = axis_idx > 1

            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df[cols["open"]],
                    high=df[cols["high"]],
                    low=df[cols["low"]],
                    close=df[cols["close"]],
                    name=f"{df.symbol.iloc[-1]} (Pane {pane_num})",
                    increasing_line_color="green",
                    decreasing_line_color="red",
                ),
                row=pane_num,
                col=1,
                secondary_y=use_secondary_y,
            )
            # track axis properties
            yaxes_dict[pane_num][logical_yaxis] = {
                "side": "right" if use_secondary_y else "left",
                "color": yaxis_colors[(axis_idx - 1) % len(yaxis_colors)],
                "title": "Price",
            }
            # update range on correct axis
            low_val, high_val = df[cols["low"]].min(), df[cols["high"]].max()
            key = "secondary" if use_secondary_y else "primary"
            axis_ranges[pane_num][key] = [
                min(axis_ranges[pane_num][key][0], low_val),
                max(axis_ranges[pane_num][key][1], high_val),
            ]

    # 2. Overlay indicator columns per pane
    if df_features:
        for pane_num, overlays in df_features.items():
            for overlay in overlays:
                # Extract df_idx and column information
                df_idx = overlay.get("df_idx", 1)  # Default to the first DataFrame if df_idx is not provided
                col = overlay["column"] if isinstance(overlay, dict) else overlay
                logical_yaxis = overlay.get("yaxis", "y") if isinstance(overlay, dict) else "y"

                # Validate df_idx
                if df_idx < 1 or df_idx > len(df_list):
                    get_ohlcutils_logger().log_error(f"df_idx {df_idx} out of range (1-{len(df_list)})", ValueError(f"df_idx {df_idx} out of range (1-{len(df_list)})"), {
                        "df_idx": df_idx,
                        "df_list_length": len(df_list),
                        "valid_range": f"1-{len(df_list)}",
                        "function": "plot"
                    })
                    raise ValueError(f"df_idx {df_idx} out of range (1-{len(df_list)})")

                # Select the appropriate DataFrame
                df = df_list[df_idx - 1]
                if col not in df.columns:
                    get_ohlcutils_logger().log_error(f"Column '{col}' not found in DataFrame {df_idx}", KeyError(f"Column '{col}' not found in DataFrame {df_idx}"), {
                        "column": col,
                        "df_idx": df_idx,
                        "available_columns": list(df.columns),
                        "function": "plot"
                    })
                    raise KeyError(f"Column '{col}' not found in DataFrame {df_idx}")

                # Determine y-axis
                axis_idx = int(logical_yaxis[1:]) if logical_yaxis != "y" else 1
                use_secondary_y = axis_idx > 1

                # Add feature trace
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        name=f"{col} (Pane {pane_num})",
                        mode="lines",
                        line=dict(
                            color=yaxis_colors[(axis_idx - 1) % len(yaxis_colors)],
                            dash="dot" if df[col].nunique() / len(df[col]) < 0.5 else "solid",
                        ),
                    ),
                    row=pane_num,
                    col=1,
                    secondary_y=use_secondary_y,
                )

                # Update axis properties
                yaxes_dict[pane_num][logical_yaxis] = {
                    "side": "right" if use_secondary_y else "left",
                    "color": yaxis_colors[(axis_idx - 1) % len(yaxis_colors)],
                    "title": col,
                }

                # Update range
                key = "secondary" if use_secondary_y else "primary"
                min_val, max_val = df[col].min(), df[col].max()
                axis_ranges[pane_num][key] = [
                    min(axis_ranges[pane_num][key][0], min_val),
                    max(axis_ranges[pane_num][key][1], max_val),
                ]

    # 3. Calculate and plot pane indicators (all panes)
    if ta_indicators:
        for pane_num, indicators in ta_indicators.items():
            for indicator in indicators:
                name = indicator.get("name")
                kwargs = indicator.get("kwargs", {})
                column_name = indicator.get("column_name", name)
                df_idx = indicator.get("df_idx", 1)
                df = df_list[df_idx - 1]
                logical_yaxis = indicator.get("yaxis", "y")
                axis_idx = int(logical_yaxis[1:]) if logical_yaxis != "y" else 0
                # For Plotly with secondary_y, we can only have primary (False) or secondary (True)
                use_secondary_y = axis_idx > 1
                axis_type = "secondary" if use_secondary_y else "primary"

                # Calculate indicator
                if hasattr(ta, name):
                    ta_function = getattr(ta, name)

                    # Map kwargs values to the corresponding DataFrame columns
                    mapped_kwargs = {k: (df[v] if v in df else v) for k, v in kwargs.items()}

                    # Pass the mapped kwargs to the pandas-ta function
                    result = ta_function(**mapped_kwargs, append=False)  # Ensure the function returns the result

                    if result is None:
                        get_ohlcutils_logger().log_error(f"Indicator '{name}' did not return a result. Check the arguments: {kwargs}", ValueError(f"Indicator '{name}' did not return a result. Check the arguments: {kwargs}"), {
                            "indicator_name": name,
                            "kwargs": kwargs,
                            "function": "plot"
                        })
                        raise ValueError(f"Indicator '{name}' did not return a result. Check the arguments: {kwargs}")

                    # Update axis ranges and add traces for each column in the result
                    if isinstance(result, pd.DataFrame):
                        columns_to_plot = indicator.get("columns", result.columns)
                        for col in columns_to_plot:
                            full_column_name = f"{column_name}_{col}"
                            df[full_column_name] = result[col]

                            # Update axis ranges based on data
                            min_val = df[full_column_name].min()
                            max_val = df[full_column_name].max()
                            axis_ranges[pane_num][axis_type] = [
                                min(axis_ranges[pane_num][axis_type][0], min_val),
                                max(axis_ranges[pane_num][axis_type][1], max_val),
                            ]

                            # Define a color palette
                            color_palette = (
                                plotly.colors.qualitative.Plotly
                            )  # Use Plotly's default qualitative color palette
                            color_count = len(color_palette)  # Number of colors in the palette
                            indicator_color_map: Dict[str, str] = {}  # Map to store assigned colors for each indicator

                            # Add trace for each column
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df[full_column_name],
                                    name=f"{full_column_name} (Pane {pane_num})",
                                    mode="lines",
                                    line=dict(
                                        color=indicator_color_map.setdefault(
                                            column_name, color_palette[len(indicator_color_map) % color_count]
                                        ),
                                        dash=(
                                            "dot"
                                            if df[full_column_name].nunique() / len(df[full_column_name]) < 0.5
                                            else "solid"
                                        ),
                                    ),
                                ),
                                row=pane_num,
                                col=1,
                                secondary_y=use_secondary_y,
                            )
                            # Update y-axis dictionary
                            if logical_yaxis not in yaxes_dict[pane_num]:
                                yaxes_dict[pane_num][logical_yaxis] = {
                                    "side": "right" if use_secondary_y else "left",
                                    "color": yaxis_colors[(axis_idx - 1) % 4],
                                    "title": full_column_name,
                                }
                    else:
                        # Handle single Series result
                        df[column_name] = result
                        min_val = df[column_name].min()
                        max_val = df[column_name].max()
                        axis_ranges[pane_num][axis_type] = [
                            min(axis_ranges[pane_num][axis_type][0], min_val),
                            max(axis_ranges[pane_num][axis_type][1], max_val),
                        ]

                        # Add trace for the single column
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df[column_name],
                                name=column_name if pane_num == 1 else f"{column_name} (Pane {pane_num})",
                                mode="lines",
                                line=dict(color=yaxis_colors[(axis_idx - 1) % 4]),
                            ),
                            row=pane_num,
                            col=1,
                            secondary_y=use_secondary_y,
                        )

                        # Update y-axis dictionary
                        if logical_yaxis not in yaxes_dict[pane_num]:
                            yaxes_dict[pane_num][logical_yaxis] = {
                                "side": "right" if use_secondary_y else "left",
                                "color": yaxis_colors[(axis_idx - 1) % 4],
                                "title": column_name,
                            }

    # --- X-axis labels ---
    x_labels = df_list[0].index.strftime("%Y-%m-%d %H:%M:%S").tolist()
    x_labels = [label.split(" ")[0] if label.endswith("00:00:00") else label for label in x_labels]
    total_points = len(x_labels)
    step = max(1, total_points // (max_x_labels - 1))
    selected_indices = sorted(set(list(range(0, total_points, step)) + [total_points - 1]))
    x_tickvals = [df_list[0].index[i] for i in selected_indices]
    x_ticktext = [x_labels[i] for i in selected_indices]

    symbol = df_list[0]["symbol"].iloc[0] if "symbol" in df_list[0].columns else ""

    # --- Update axes properties for each pane ---
    for pane in range(1, n_panes + 1):
        # Primary y-axis (left side)
        left_title = "Price" if pane == 1 else "Value"
        if pane in yaxes_dict and "y" in yaxes_dict[pane]:
            left_title = yaxes_dict[pane]["y"].get("title", left_title)

        # Add some padding to the ranges (5%)
        if axis_ranges[pane]["primary"][0] != float("inf"):
            p_range = axis_ranges[pane]["primary"]
            range_size = p_range[1] - p_range[0]
            padding = range_size * 0.05
            primary_range = [p_range[0] - padding, p_range[1] + padding]

            fig.update_yaxes(
                title=dict(text=left_title, font=dict(color=yaxis_colors[0])),
                tickfont=dict(color=yaxis_colors[0]),
                showgrid=True,
                zeroline=False,
                range=primary_range,  # Set explicit range for the axis
                row=pane,
                col=1,
                secondary_y=False,
            )
        else:
            fig.update_yaxes(
                title=dict(text=left_title, font=dict(color=yaxis_colors[0])),
                tickfont=dict(color=yaxis_colors[0]),
                showgrid=True,
                zeroline=False,
                row=pane,
                col=1,
                secondary_y=False,
            )

        # Secondary y-axis (right side) - only update if we have indicators using it
        has_secondary = "y2" in yaxes_dict.get(pane, {})  # fix: it could be y2 or y3...does it matter?
        if has_secondary:
            right_title = yaxes_dict[pane].get("y2", {}).get("title", "")

            # Add some padding to the ranges (5%)
            if axis_ranges[pane]["secondary"][0] != float("inf"):
                s_range = axis_ranges[pane]["secondary"]
                range_size = s_range[1] - s_range[0]
                padding = range_size * 0.05
                secondary_range = [s_range[0] - padding, s_range[1] + padding]

                fig.update_yaxes(
                    title=dict(text=right_title, font=dict(color=yaxis_colors[1])),
                    tickfont=dict(color=yaxis_colors[1]),
                    showgrid=False,
                    zeroline=False,
                    range=secondary_range,  # Set explicit range for the axis
                    row=pane,
                    col=1,
                    secondary_y=True,
                )
            else:
                fig.update_yaxes(
                    title=dict(text=right_title, font=dict(color=yaxis_colors[1])),
                    tickfont=dict(color=yaxis_colors[1]),
                    showgrid=False,
                    zeroline=False,
                    row=pane,
                    col=1,
                    secondary_y=True,
                )

    # --- Configure x-axis ---
    fig.update_xaxes(
        type="category",
        tickvals=x_tickvals,
        ticktext=x_ticktext,
        tickangle=-90,
        row=n_panes,  # Apply to bottom-most pane
        col=1,
    )

    # Add spikes to all panes
    for i in range(1, n_panes + 1):
        fig.update_xaxes(
            type="category",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            row=i,
            col=1,
        )
        # Add spikes to both primary and secondary y-axes
        for secondary in [False, True]:
            fig.update_yaxes(
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikethickness=1,
                row=i,
                col=1,
                secondary_y=secondary,
            )

    # --- Final layout update ---
    fig.update_layout(
        title=f"{title}{' - ' + symbol if symbol else ''}",
        height=600 + (n_panes - 1) * 180,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )

    # apply computed ranges to each pane's secondary y-axis
    for pane in range(1, n_panes + 1):
        sec_min, sec_max = axis_ranges[pane]["secondary"]
        if sec_min != float("inf") and sec_max != float("-inf"):
            pad = (sec_max - sec_min) * 0.05
            fig.update_yaxes(
                range=[sec_min - pad, sec_max + pad],
                row=pane,
                col=1,
                secondary_y=True,
            )
    fig.update_xaxes(rangeslider_visible=False)
    fig.show()
    return fig

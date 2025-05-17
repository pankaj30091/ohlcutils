import pandas_ta as ta
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "notebook_connected"


def plot(
    df_list,
    candle_stick_columns={"open": "aopen", "high": "ahigh", "low": "alow", "close": "asettle", "volume": "avolume"},
    indicator_columns=None,
    ta_indicators=None,  # List of pandas-ta indicators to calculate and plot
    title="",
    max_x_labels=10,  # Maximum number of x-axis labels to display
    separate_y_axes=None,  # List of column names to plot on separate y-axes
):
    # Initialize y-axis count
    yaxis_count = 1

    """
    Plot a candlestick chart using Plotly.

    Parameters:
    - df_list: List of DataFrames containing OHLC data.
    - candle_stick_columns: Dictionary mapping DataFrame columns to OHLC keys.
    - indicator_columns: List of column names to plot from other DataFrames. If None, all columns are considered.
    - ta_indicators: List of dictionaries specifying pandas-ta indicators to calculate and plot.
                     Each dictionary should have the format:
                     {"name": "indicator_name", "kwargs": {...}, "column_name": "result_column_name", "target_column": "column_to_use"}.
    - title: Title of the chart.
    - max_x_labels: Maximum number of x-axis labels to display (default is 10).
    - separate_y_axes: List of column names to plot on separate y-axes (default is None).
    """
    fig = go.Figure()

    # Plot the candlestick chart from the first DataFrame
    main_df = df_list[0]
    fig.add_trace(
        go.Candlestick(
            x=main_df.index,
            open=main_df[candle_stick_columns["open"]],
            high=main_df[candle_stick_columns["high"]],
            low=main_df[candle_stick_columns["low"]],
            close=main_df[candle_stick_columns["close"]],
            name="Candles",
            increasing_line_color="green",
            decreasing_line_color="red",
        )
    )

    # Determine the range of the candlestick chart
    candlestick_min = main_df[candle_stick_columns["low"]].min()
    candlestick_max = main_df[candle_stick_columns["high"]].max()

    # Calculate and add pandas-ta indicators
    if ta_indicators:
        for indicator in ta_indicators:
            name = indicator.get("name")
            kwargs = indicator.get("kwargs", {})
            column_name = indicator.get("column_name", name)
            target_column = indicator.get("target_column", candle_stick_columns["close"])  # Default to 'close'

            # Calculate the indicator using pandas-ta
            if hasattr(ta, name):
                main_df[column_name] = getattr(ta, name)(main_df[target_column], **kwargs)
            else:
                raise ValueError(f"Indicator '{name}' is not available in pandas-ta.")

            # Add the indicator to the plot
            if separate_y_axes and column_name in separate_y_axes:
                yaxis_count += 1
                yaxis = f"y{yaxis_count}"
                fig.add_trace(
                    go.Scatter(
                        x=main_df.index,
                        y=main_df[column_name],
                        name=f"{column_name}",
                        mode="lines",
                        yaxis=yaxis,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=main_df.index,
                        y=main_df[column_name],
                        name=f"Indicator - {column_name}",
                        mode="lines",
                    )
                )

    # Plot additional DataFrames
    for i, df in enumerate(df_list[1:], start=2):
        # Exclude candlestick columns if indicator_columns is None
        columns_to_plot = (
            indicator_columns
            if indicator_columns
            else [col for col in df.columns if col not in candle_stick_columns.values()]
        )
        for col in columns_to_plot:
            if col not in df.columns:  # Check if the column exists in the DataFrame
                continue

            col_min = df[col].min()
            col_max = df[col].max()

            # Check if the column's range overlaps with the candlestick range
            if separate_y_axes and col in separate_y_axes:
                yaxis_count += 1
                yaxis = f"y{yaxis_count}"  # Create a new y-axis
            elif candlestick_min <= col_min <= candlestick_max or candlestick_min <= col_max <= candlestick_max:
                yaxis = "y1"  # Plot on the same axis as the candlestick chart
            else:
                yaxis_count += 1
                yaxis = f"y{yaxis_count}"  # Create a new y-axis

            # Add the trace
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=f"{col}",
                    yaxis=yaxis,
                    mode="lines",
                )
            )

    # Simplify x-axis labels
    x_labels = main_df.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
    x_labels = [
        label.split(" ")[0] if label.endswith("00:00:00") else label for label in x_labels
    ]  # Remove time if it's midnight

    # Select evenly spaced labels, including the first and last
    total_points = len(x_labels)
    step = max(1, total_points // (max_x_labels - 1))
    selected_indices = list(range(0, total_points, step)) + [total_points - 1]
    selected_indices = sorted(set(selected_indices))  # Ensure unique and sorted indices
    x_tickvals = [main_df.index[i] for i in selected_indices]
    x_ticktext = [x_labels[i] for i in selected_indices]

    # Get the first value of the symbol column from the first DataFrame
    symbol = main_df["symbol"].iloc[0] if "symbol" in main_df.columns else ""

    # Update layout for multiple y-axes
    layout_yaxes = {
        "yaxis": dict(
            title="Price",
            side="left",
            showspikes=True,  # Enable horizontal spikeline
            spikemode="across",  # Spikeline across all subplots
            spikesnap="cursor",  # Snap spikeline to cursor
            spikethickness=1,
        ),
    }
    for j in range(2, yaxis_count + 1):
        layout_yaxes[f"yaxis{j}"] = dict(
            title=f"Axis {j}",
            overlaying="y",
            side="right" if j % 2 == 0 else "left",
            showgrid=False,
            showspikes=True,  # Enable horizontal spikeline for additional y-axes
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
        )
        if j > 2:  # For additional axes beyond the first overlay
            layout_yaxes[f"yaxis{j}"]["anchor"] = "free"
            layout_yaxes[f"yaxis{j}"]["position"] = (j - 2) / (yaxis_count + 1)  # Normalize position to [0, 1]

    fig.update_layout(
        title=f"{title}{symbol}",
        xaxis=dict(
            type="category",
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            tickangle=-90,  # Rotate x-axis labels vertically
            rangeslider=dict(visible=False),
            showspikes=True,  # Enable vertical spikeline
            spikemode="across",  # Spikeline across all subplots
            spikesnap="cursor",  # Snap spikeline to cursor
            spikethickness=1,
        ),
        hovermode="x unified",  # Unified tooltips with crosshair
        height=600,
        **layout_yaxes,  # Use layout_yaxes to define all y-axes
    )

    fig.show()

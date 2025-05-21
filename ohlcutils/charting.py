import pandas as pd
import pandas_ta as ta
import plotly.colors
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .config import get_config


def get_dynamic_config():
    return get_config()


pio.renderers.default = get_dynamic_config().get("chart_rendering")


def plot(
    df_list,
    candle_stick_columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"},
    df_features=None,  # Dict: {df_idx: [ { "column": col, "yaxis": "y2" }, ... ] }
    ta_indicators=None,  # Dict: {pane_number: [indicator_dicts]}
    title="",
    max_x_labels=10,
    pane_titles=None,
    max_yaxes_per_pane=4,
    pane_heights=None,  # New parameter to define pane heights
):
    """
    Multi-pane candlestick and indicator plot using Plotly, supporting up to 4 y-axes per pane.

    - indicator_columns: Dict {df_idx: [ { "column": col, "yaxis": "y2" }, ... ] }
       (e.g. {1: [{"column": "trend", "yaxis": "y2"}]})
    - pane_indicators: indicators can specify "yaxis": "y2", "y3", ...
    - pane_heights: List of relative heights for each pane (e.g., [0.6, 0.2, 0.2]).
    """
    n_panes = max(ta_indicators.keys()) if ta_indicators else 1

    # Calculate row heights
    if pane_heights:
        if len(pane_heights) != n_panes:
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

    main_df = df_list[0]

    # --- Track y-axes per pane ---
    yaxes_dict = {pane: {} for pane in range(1, n_panes + 1)}
    yaxis_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Track min/max values for each axis to set proper ranges
    axis_ranges = {
        pane: {"primary": [float("inf"), float("-inf")], "secondary": [float("inf"), float("-inf")]}
        for pane in range(1, n_panes + 1)
    }

    # 1. Candlestick in main pane (row=1) - always on y (primary left axis)
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
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    yaxes_dict[1]["y"] = {"side": "left", "color": yaxis_colors[0], "title": "Price"}

    # Update price range for primary axis in pane 1
    low_values = main_df[candle_stick_columns["low"]].min()
    high_values = main_df[candle_stick_columns["high"]].max()
    axis_ranges[1]["primary"] = [
        min(axis_ranges[1]["primary"][0], low_values),
        max(axis_ranges[1]["primary"][1], high_values),
    ]

    # 2. Overlay indicator columns (main pane) with axis selection
    if df_features:
        for dfi, overlays in df_features.items():  # change: dfi should be the pane on which indicators are plotted
            df = df_list[dfi]  # identify df from df_list. df should contain column name == overlays["column"]
            for overlay in overlays:
                col = overlay["column"] if isinstance(overlay, dict) else overlay
                logical_yaxis = overlay.get("yaxis", "y") if isinstance(overlay, dict) else "y"
                axis_idx = int(logical_yaxis[1:]) if logical_yaxis != "y" else 1
                if axis_idx > max_yaxes_per_pane:
                    raise ValueError(f"Max {max_yaxes_per_pane} y-axes per pane supported.")

                # For Plotly with secondary_y, we can only have primary (False) or secondary (True)
                use_secondary_y = axis_idx > 1

                if col in df.columns:
                    # Update axis ranges based on data
                    axis_type = "secondary" if use_secondary_y else "primary"
                    min_val = df[col].min()
                    max_val = df[col].max()
                    axis_ranges[1][axis_type] = [
                        min(axis_ranges[1][axis_type][0], min_val),
                        max(axis_ranges[1][axis_type][1], max_val),
                    ]

                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[col],
                            name=f"{col}" if dfi == 0 else f"{col} (df{dfi})",
                            mode="lines",
                            line=dict(
                                color=yaxis_colors[(axis_idx - 1) % 4],
                                dash="dot" if df[col].nunique() / len(df[col]) < 0.5 else "solid",
                            ),
                        ),
                        row=1,  # question: what is impact of this row?
                        col=1,
                        secondary_y=use_secondary_y,
                    )
                    if logical_yaxis not in yaxes_dict[1]:
                        yaxes_dict[1][logical_yaxis] = {
                            "side": "right" if use_secondary_y else "left",
                            "color": yaxis_colors[(axis_idx - 1) % 4],
                            "title": col,
                        }

    # 3. Calculate and plot pane indicators (all panes)
    if ta_indicators:
        for pane_num, indicators in ta_indicators.items():
            for indicator in indicators:
                name = indicator.get("name")
                kwargs = indicator.get("kwargs", {})
                column_name = indicator.get("column_name", name)
                df_idx = indicator.get("df_idx", 0)
                df = df_list[df_idx]
                logical_yaxis = indicator.get("yaxis", "y")
                axis_idx = int(logical_yaxis[1:]) if logical_yaxis != "y" else 0
                if axis_idx > max_yaxes_per_pane:
                    raise ValueError(f"Max {max_yaxes_per_pane} y-axes per pane supported.")

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
                            indicator_color_map = {}  # Map to store assigned colors for each indicator

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
    x_labels = main_df.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
    x_labels = [label.split(" ")[0] if label.endswith("00:00:00") else label for label in x_labels]
    total_points = len(x_labels)
    step = max(1, total_points // (max_x_labels - 1))
    selected_indices = sorted(set(list(range(0, total_points, step)) + [total_points - 1]))
    x_tickvals = [main_df.index[i] for i in selected_indices]
    x_ticktext = [x_labels[i] for i in selected_indices]

    symbol = main_df["symbol"].iloc[0] if "symbol" in main_df.columns else ""

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

    fig.show()

"""Plotly categorical plot abstraction mimicking seaborn's catplot API."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Tab10 colors matching matplotlib/seaborn's "tab10" palette
TAB10_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _get_palette_colors(palette: str | list[str], n_colors: int) -> list[str]:
    """Get a list of colors from a palette name or list.

    Parameters
    ----------
    palette : str or list of str
        Palette name (e.g. "tab10") or a list of color strings.
    n_colors : int
        Number of colors to return.

    Returns
    -------
    list of str
        List of hex color strings.
    """
    if isinstance(palette, list):
        return palette[:n_colors]
    if palette == "tab10":
        return TAB10_COLORS[:n_colors]
    # fallback to tab10
    return TAB10_COLORS[:n_colors]


class PlotlyFacetGrid:
    """Result object mimicking seaborn's FacetGrid interface for plotly.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        The plotly figure.
    data : DataFrame
        The data used to create the plot.
    col : str or None
        The column used for faceting.
    col_values : list
        The unique values of the facet column.
    hue : str or None
        The column used for color grouping.
    hue_values : list
        The unique values of the hue column.
    x : str
        The x-axis column name.
    y : str
        The y-axis column name.
    palette_colors : list of str
        Colors used for the hue groups.

    Attributes
    ----------
    figure : plotly.graph_objects.Figure
        The plotly figure.
    axes : ndarray
        Placeholder for compatibility with the matplotlib API.
    """

    def __init__(
        self,
        figure: go.Figure,
        data: pd.DataFrame,
        col: str | None,
        col_values: list,
        hue: str | None,
        hue_values: list,
        x: str,
        y: str,
        palette_colors: list[str],
    ):
        self.figure = figure
        self._data = data
        self._col = col
        self._col_values = col_values
        self._hue = hue
        self._hue_values = hue_values
        self._x = x
        self._y = y
        self._palette_colors = palette_colors
        # Create an axes-like array for compatibility
        self.axes = np.array([None] * len(col_values))

    def map_dataframe(self, func, **kwargs):
        """Apply a plotting function to each facet's data.

        This mimics seaborn's ``FacetGrid.map_dataframe`` method, allowing
        additional traces (e.g. boxplot) to be overlaid on an existing plot.

        Parameters
        ----------
        func : callable
            A function that takes (figure, data, col_idx, hue, hue_values,
            palette_colors, **kwargs) and adds traces to the figure.
        **kwargs : dict
            Additional keyword arguments passed to ``func``.

        Returns
        -------
        self
        """
        for col_idx, col_val in enumerate(self._col_values):
            if col_val is not None:
                facet_data = self._data[self._data[self._col] == col_val]
            else:
                facet_data = self._data
            func(
                self.figure,
                facet_data,
                col_idx=col_idx,
                n_cols=len(self._col_values),
                hue=self._hue,
                hue_values=self._hue_values,
                palette_colors=self._palette_colors,
                **kwargs,
            )
        return self

    def tight_layout(self):
        """No-op for plotly compatibility."""


def _add_bar_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None,
    hue_values: list,
    col_idx: int,
    n_cols: int,
    palette_colors: list[str],
) -> None:
    """Add horizontal bar traces to a plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to add traces to.
    data : DataFrame
        The data to plot.
    x : str
        Column name for the x-axis (values).
    y : str
        Column name for the y-axis (categories).
    hue : str or None
        Column name for color grouping.
    hue_values : list
        Unique values of the hue column.
    col_idx : int
        The subplot column index (0-based).
    n_cols : int
        Total number of subplot columns.
    palette_colors : list of str
        Colors for the hue groups.
    """
    show_legend = col_idx == 0

    if hue is not None:
        for color_idx, hue_val in enumerate(hue_values):
            hue_data = data[data[hue] == hue_val]
            fig.add_trace(
                go.Bar(
                    x=hue_data[x],
                    y=hue_data[y],
                    orientation="h",
                    name=str(hue_val),
                    marker_color=palette_colors[color_idx % len(palette_colors)],
                    legendgroup=str(hue_val),
                    showlegend=show_legend,
                ),
                row=1,
                col=col_idx + 1,
            )
    else:
        fig.add_trace(
            go.Bar(
                x=data[x],
                y=data[y],
                orientation="h",
                marker_color=palette_colors[0],
                showlegend=False,
            ),
            row=1,
            col=col_idx + 1,
        )


def _add_strip_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None,
    hue_values: list,
    col_idx: int,
    n_cols: int,
    palette_colors: list[str],
    alpha: float = 0.5,
    dodge: bool = True,
) -> None:
    """Add strip plot (jittered scatter) traces to a plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to add traces to.
    data : DataFrame
        The data to plot.
    x : str
        Column name for the x-axis (values).
    y : str
        Column name for the y-axis (categories).
    hue : str or None
        Column name for color grouping.
    hue_values : list
        Unique values of the hue column.
    col_idx : int
        The subplot column index (0-based).
    n_cols : int
        Total number of subplot columns.
    palette_colors : list of str
        Colors for the hue groups.
    alpha : float, default=0.5
        Opacity of the strip points.
    dodge : bool, default=True
        Whether to dodge the points by hue group.
    """
    show_legend = col_idx == 0

    if hue is not None:
        for color_idx, hue_val in enumerate(hue_values):
            hue_data = data[data[hue] == hue_val]
            fig.add_trace(
                go.Scatter(
                    x=hue_data[x],
                    y=hue_data[y],
                    mode="markers",
                    name=str(hue_val),
                    marker={
                        "color": palette_colors[color_idx % len(palette_colors)],
                        "opacity": alpha,
                        "size": 6,
                    },
                    legendgroup=str(hue_val),
                    showlegend=show_legend,
                ),
                row=1,
                col=col_idx + 1,
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=data[x],
                y=data[y],
                mode="markers",
                marker={
                    "color": palette_colors[0],
                    "opacity": alpha,
                    "size": 6,
                },
                showlegend=False,
            ),
            row=1,
            col=col_idx + 1,
        )


def add_boxplot(
    fig: go.Figure,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None,
    hue_values: list,
    col_idx: int,
    n_cols: int,
    palette_colors: list[str],
    dodge: bool = True,
    whis: float = 1.5,
    **kwargs: Any,
) -> None:
    """Add box plot traces to an existing plotly figure.

    This function is designed to be used with
    :meth:`PlotlyFacetGrid.map_dataframe` to overlay box plots on strip plots,
    mimicking the seaborn pattern of ``facet.map_dataframe(sns.boxplot, ...)``.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to add traces to.
    data : DataFrame
        The data to plot.
    x : str
        Column name for the x-axis (values).
    y : str
        Column name for the y-axis (categories).
    hue : str or None
        Column name for color grouping.
    hue_values : list
        Unique values of the hue column.
    col_idx : int
        The subplot column index (0-based).
    n_cols : int
        Total number of subplot columns.
    palette_colors : list of str
        Colors for the hue groups.
    dodge : bool, default=True
        Whether to dodge the boxes by hue group.
    whis : float, default=1.5
        Whisker length as a multiple of IQR. Very large values effectively
        extend whiskers to data range (matching seaborn's ``whis=1e10``).
    **kwargs : dict
        Additional keyword arguments (ignored, for API compatibility).
    """
    if hue is not None:
        for color_idx, hue_val in enumerate(hue_values):
            hue_data = data[data[hue] == hue_val]
            color = palette_colors[color_idx % len(palette_colors)]
            for feature_val in hue_data[y].unique():
                feature_data = hue_data[hue_data[y] == feature_val]
                fig.add_trace(
                    go.Box(
                        x=feature_data[x],
                        y=feature_data[y],
                        orientation="h",
                        name=str(hue_val),
                        marker_color=color,
                        line_color="black",
                        fillcolor="rgba(0,0,0,0)",
                        legendgroup=str(hue_val),
                        showlegend=False,
                        boxpoints=False,
                        whiskerwidth=0.5,
                    ),
                    row=1,
                    col=col_idx + 1,
                )
    else:
        for feature_val in data[y].unique():
            feature_data = data[data[y] == feature_val]
            fig.add_trace(
                go.Box(
                    x=feature_data[x],
                    y=feature_data[y],
                    orientation="h",
                    marker_color=palette_colors[0],
                    line_color="black",
                    fillcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    boxpoints=False,
                    whiskerwidth=0.5,
                ),
                row=1,
                col=col_idx + 1,
            )


def _decorate_plotly_axis(
    fig: go.Figure,
    *,
    col_idx: int,
    n_cols: int,
    add_background_features: bool,
    feature_names: list[str],
    xlabel: str,
    ylabel: str,
) -> None:
    """Decorate a plotly subplot axis.

    This is the plotly equivalent of
    :func:`skore._sklearn._plot.inspection.utils._decorate_matplotlib_axis`.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to decorate.
    col_idx : int
        The subplot column index (0-based).
    n_cols : int
        Total number of subplot columns.
    add_background_features : bool
        Whether to add alternating background bands for features.
    feature_names : list of str
        The feature names displayed on the y-axis.
    xlabel : str
        The x-axis label.
    ylabel : str
        The y-axis label.
    """
    # Axis suffix for subplot referencing
    axis_suffix = "" if col_idx == 0 else str(col_idx + 1)

    # Add vertical reference line at x=0
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        row=1,
        col=col_idx + 1,
    )

    # Update axes styling
    fig.update_layout(
        **{
            f"xaxis{axis_suffix}": {
                "title_text": xlabel,
                "showgrid": True,
                "gridcolor": "lightgray",
                "zeroline": False,
            },
            f"yaxis{axis_suffix}": {
                "title_text": ylabel,
                "showgrid": False,
            },
        }
    )

    # Add alternating background bands for features
    if add_background_features:
        for feature_idx in range(0, len(feature_names), 2):
            fig.add_hrect(
                y0=feature_idx - 0.5,
                y1=feature_idx + 0.5,
                fillcolor="lightgray",
                opacity=0.2,
                line_width=0,
                row=1,
                col=col_idx + 1,
            )


def catplot(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None = None,
    col: str | None = None,
    kind: Literal["bar", "strip"] = "bar",
    dodge: bool = True,
    height: float = 6,
    aspect: float = 2,
    palette: str | list[str] = "tab10",
    alpha: float | None = None,
    sharey: bool = True,
    **kwargs: Any,
) -> PlotlyFacetGrid:
    """Create a categorical plot using plotly, mimicking seaborn's catplot API.

    Parameters
    ----------
    data : DataFrame
        The data to plot.
    x : str
        Column name for the x-axis (values).
    y : str
        Column name for the y-axis (categories).
    hue : str or None, default=None
        Column name for color grouping.
    col : str or None, default=None
        Column name for faceting into subplots.
    kind : {"bar", "strip"}, default="bar"
        The kind of categorical plot.
    dodge : bool, default=True
        Whether to dodge grouped elements.
    height : float, default=6
        Height of each facet (in seaborn units, converted to pixels).
    aspect : float, default=2
        Aspect ratio of each facet (width = height * aspect).
    palette : str or list of str, default="tab10"
        Color palette name or list of colors.
    alpha : float or None, default=None
        Opacity for strip plot markers.
    sharey : bool, default=True
        Whether subplots share the y-axis.
    **kwargs : dict
        Additional keyword arguments (for forward compatibility).

    Returns
    -------
    PlotlyFacetGrid
        A wrapper around the plotly Figure with faceting metadata.
    """
    col_values = data[col].unique().tolist() if col else [None]
    n_cols = len(col_values)
    n_hue = data[hue].nunique() if hue else 1
    hue_values = data[hue].unique().tolist() if hue else []
    palette_colors = _get_palette_colors(palette, n_hue)

    # Create subplots
    subplot_titles = [f"{col} = {v}" for v in col_values] if col else None
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_yaxes=sharey,
        horizontal_spacing=0.05 if n_cols > 1 else 0,
    )

    # Convert seaborn size conventions to plotly pixels
    # seaborn height is in inches, aspect is width/height ratio
    dpi = 80
    height_px = int(height * dpi)
    width_px = int(height * aspect * dpi * n_cols)
    fig.update_layout(
        width=width_px,
        height=height_px,
        barmode="group" if dodge else "overlay",
        template="plotly_white",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
        },
    )

    add_func = _add_bar_traces if kind == "bar" else _add_strip_traces
    extra_kwargs = {}
    if kind == "strip" and alpha is not None:
        extra_kwargs["alpha"] = alpha

    for col_idx, col_val in enumerate(col_values):
        facet_data = data[data[col] == col_val] if col else data
        add_func(
            fig,
            facet_data,
            x=x,
            y=y,
            hue=hue,
            hue_values=hue_values,
            col_idx=col_idx,
            n_cols=n_cols,
            palette_colors=palette_colors,
            **extra_kwargs,
        )

    return PlotlyFacetGrid(
        figure=fig,
        data=data,
        col=col,
        col_values=col_values,
        hue=hue,
        hue_values=hue_values,
        x=x,
        y=y,
        palette_colors=palette_colors,
    )

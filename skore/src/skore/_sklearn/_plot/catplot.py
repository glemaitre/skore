"""Backend-dispatching categorical plot functions.

Provides a seaborn-like ``catplot`` API that dispatches to seaborn (matplotlib)
or plotly depending on ``configuration.plot_backend``. This module is the single
place where the backend switch happens for categorical plots, so that display
classes stay backend-agnostic.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from skore import configuration
from skore._sklearn.types import PlotBackend


def catplot(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None = None,
    col: str | None = None,
    kind: Literal["bar", "strip"] = "bar",
    dodge: bool = True,
    **kwargs: Any,
):
    """Create a categorical plot using the configured backend.

    The signature mirrors :func:`seaborn.catplot`. For the plotly backend only
    a subset of keyword arguments is forwarded (``height``, ``aspect``,
    ``palette``, ``alpha``, ``sharey``); the rest is silently ignored so that
    callers can pass the full seaborn kwargs without worrying about the active
    backend.

    Parameters
    ----------
    data : DataFrame
        Long-form data for plotting.
    x, y : str
        Column names for the axes.
    hue : str or None
        Column for colour grouping.
    col : str or None
        Column for faceting into subplots.
    kind : {"bar", "strip"}
        Type of categorical plot.
    dodge : bool
        Whether to separate hue levels along the categorical axis.
    **kwargs
        Forwarded to the underlying backend (seaborn or plotly).

    Returns
    -------
    seaborn.FacetGrid or PlotlyFacetGrid
    """
    backend: PlotBackend = configuration.plot_backend

    if backend == "matplotlib":
        import seaborn as sns

        return sns.catplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            col=col,
            kind=kind,
            dodge=dodge,
            **kwargs,
        )
    elif backend == "plotly":
        from skore._sklearn._plot._plotly.catplot import catplot as _plotly_catplot

        _PLOTLY_KWARGS = {"height", "aspect", "palette", "alpha", "sharey"}
        plotly_kwargs = {k: v for k, v in kwargs.items() if k in _PLOTLY_KWARGS}

        return _plotly_catplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            col=col,
            kind=kind,
            dodge=dodge,
            **plotly_kwargs,
        )

    raise NotImplementedError(
        f"Plotting backend {backend!r} is not supported. "
        f"Available options are {PlotBackend.__args__}."
    )


def overlay_boxplot(
    facet: Any,
    *,
    x: str,
    y: str,
    hue: str | None = None,
    boxplot_kwargs: dict[str, Any],
) -> Any:
    """Overlay a boxplot on an existing categorical ``facet`` grid.

    For the matplotlib backend this calls
    ``facet.map_dataframe(sns.boxplot, …)``.  For plotly it calls
    ``facet.map_dataframe(add_boxplot, …)``.

    Parameters
    ----------
    facet : seaborn.FacetGrid or PlotlyFacetGrid
        The facet grid returned by :func:`catplot`.
    x, y : str
        Column names matching those used in :func:`catplot`.
    hue : str or None
        Column for colour grouping.
    boxplot_kwargs : dict
        Backend-specific keyword arguments.  For matplotlib every key is
        forwarded to :func:`seaborn.boxplot`.  For plotly only ``whis`` is
        used.

    Returns
    -------
    facet
        The same ``facet`` object, for chaining.
    """
    backend: PlotBackend = configuration.plot_backend

    if backend == "matplotlib":
        import seaborn as sns

        palette = "tab10" if hue is not None else None
        facet.map_dataframe(
            sns.boxplot,
            x=x,
            y=y,
            hue=hue,
            palette=palette,
            dodge=True,
            **boxplot_kwargs,
        )
    elif backend == "plotly":
        from skore._sklearn._plot._plotly.catplot import add_boxplot

        whis = boxplot_kwargs.get("whis", 1e10)
        facet.map_dataframe(add_boxplot, x=x, y=y, dodge=True, whis=whis)
    else:
        raise NotImplementedError(
            f"Plotting backend {backend!r} is not supported. "
            f"Available options are {PlotBackend.__args__}."
        )

    return facet


def set_figure_title(figure: Any, title: str) -> None:
    """Set *title* on *figure* using the configured backend.

    Parameters
    ----------
    figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure object.
    title : str
        The title text.
    """
    backend: PlotBackend = configuration.plot_backend

    if backend == "matplotlib":
        figure.suptitle(title)
    elif backend == "plotly":
        figure.update_layout(title_text=title)

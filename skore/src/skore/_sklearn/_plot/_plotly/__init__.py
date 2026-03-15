"""Plotly-based plotting abstractions mimicking seaborn's API."""

from skore._sklearn._plot._plotly.catplot import PlotlyFacetGrid, catplot

__all__ = ["PlotlyFacetGrid", "catplot"]

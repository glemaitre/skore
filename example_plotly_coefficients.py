"""Small example to test the plotly implementation of CoefficientsDisplay.

Run with:
    python example_plotly_coefficients.py
"""

# %%
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.datasets import make_regression

from skore import (
    EstimatorReport,
    CrossValidationReport,
    ComparisonReport,
    configuration,
    train_test_split,
)

configuration.plot_backend = "plotly"

# %% [markdown]
# ## 1. EstimatorReport — multiclass classification

# %%
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
y = iris.target_names[y]

split = train_test_split(X=X, y=y, random_state=0, as_dict=True, shuffle=True)
report = EstimatorReport(LogisticRegression(), **split)
display = report.inspection.coefficients()
display.plot()

# %% [markdown]
# ## 2. EstimatorReport — regression

# %%
X, y = make_regression(n_samples=200, n_features=6, random_state=0)
split = train_test_split(X=X, y=y, random_state=0, as_dict=True)
report = EstimatorReport(Ridge(), **split)
display = report.inspection.coefficients()
display.plot(sorting_order="descending")

# %% [markdown]
# ## 3. CrossValidationReport — multiclass classification (strip + box)

# %%
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
y = iris.target_names[y]

report = CrossValidationReport(LogisticRegression(), X=X, y=y, splitter=5)
display = report.inspection.coefficients()
display.plot()

# %% [markdown]
# ## 4. ComparisonReport — two estimators

# %%
split = train_test_split(X=X, y=y, random_state=0, as_dict=True, shuffle=True)
r1 = EstimatorReport(LogisticRegression(C=1), **split)
r2 = EstimatorReport(LogisticRegression(C=0.01), **split)
comp = ComparisonReport([r1, r2])
display = comp.inspection.coefficients()
display.plot()

# %% [markdown]
# ## 5. subplot_by="label"

# %%
display.plot(subplot_by="label")

# %%

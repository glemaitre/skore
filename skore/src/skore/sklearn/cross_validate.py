"""cross_validate function.

This function implements a wrapper over scikit-learn's
`cross_validate <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate>`_
function in order to enrich it with more information and enable more analysis.
"""

import contextlib
import inspect
import time
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils._indexing import _safe_indexing
from sklearn.utils._response import _check_response_method, _get_response_values
from sklearn.utils.metaestimators import available_if

from skore.item.cross_validation_item import (
    CrossValidationAggregationItem,
    CrossValidationItem,
)
from skore.project import Project
from skore.sklearn._plot import RocCurveDisplay


def _find_ml_task(
    estimator, y
) -> Literal[
    "binary-classification",
    "multiclass-classification",
    "regression",
    "clustering",
    "unknown",
]:
    """Guess the ML task being addressed based on an estimator and a target array.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        An estimator.
    y : numpy.ndarray
        A target vector.

    Returns
    -------
    Literal["classification", "regression", "clustering", "unknown"]
        The guess of the kind of ML task being performed.
    """
    import sklearn.utils.multiclass
    from sklearn.base import is_classifier, is_regressor

    if y is None:
        # NOTE: The task might not be clustering
        return "clustering"

    if is_regressor(estimator):
        return "regression"

    type_of_target = sklearn.utils.multiclass.type_of_target(y)

    if is_classifier(estimator):
        if type_of_target == "binary":
            return "binary-classification"

        if type_of_target == "multiclass":
            return "multiclass-classification"

    if type_of_target == "unknown":
        return "unknown"

    if "continuous" in type_of_target:
        return "regression"

    return "classification"


def _get_scorers_to_add(estimator, y) -> list[str]:
    """Get a list of scorers based on `estimator` and `y`.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        An estimator.
    y : numpy.ndarray
        A target vector.

    Returns
    -------
    scorers_to_add : list[str]
        A list of scorers
    """
    ml_task = _find_ml_task(estimator, y)

    # Add scorers based on the ML task
    if ml_task == "regression":
        return ["r2", "neg_root_mean_squared_error"]
    if ml_task == "binary-classification":
        return ["roc_auc", "neg_brier_score", "recall", "precision"]
    if ml_task == "multiclass-classification":
        if hasattr(estimator, "predict_proba"):
            return [
                "recall_weighted",
                "precision_weighted",
                "roc_auc_ovr_weighted",
                "neg_log_loss",
            ]
        return ["recall_weighted", "precision_weighted"]
    return []


def _add_scorers(scorers, scorers_to_add):
    """Expand `scorers` with more scorers.

    The type of the resulting scorers object is dependent on the type of the input
    scorers:
    - If `scorers` is a dict, then extra scorers are added to the dict;
    - If `scorers` is a string or None, then it is converted to a dict and extra scorers
    are added to the dict;
    - If `scorers` is a list or tuple, then it is converted to a dict and extra scorers
    are added to the dict;
    - If `scorers` is a callable, then a new callable is created that
    returns a dict with the user-defined score as well as the scorers to add.
    In case the user-defined dict contains a metric with a name conflicting with the
    metrics we add, the user-defined metric always wins.

    Parameters
    ----------
    scorers : any type that is accepted by scikit-learn's cross_validate
        The scorer(s) to expand.
    scorers_to_add : list[str]
        The scorers to be added.

    Returns
    -------
    new_scorers : dict or callable
        The scorers after adding `scorers_to_add`.
    added_scorers : Iterable[str]
        The scorers that were actually added (i.e. the ones that were not already
        in `scorers`).
    """
    if scorers is None or isinstance(scorers, str):
        new_scorers, added_scorers = _add_scorers({"score": scorers}, scorers_to_add)
    elif isinstance(scorers, (list, tuple)):
        new_scorers, added_scorers = _add_scorers(
            {s: s for s in scorers}, scorers_to_add
        )
    elif isinstance(scorers, dict):
        new_scorers = {s: s for s in scorers_to_add} | scorers
        added_scorers = set(scorers_to_add) - set(scorers)
    elif callable(scorers):
        from sklearn.metrics import check_scoring
        from sklearn.metrics._scorer import _MultimetricScorer

        internal_scorer = _MultimetricScorer(
            scorers={
                s: check_scoring(estimator=None, scoring=s) for s in scorers_to_add
            }
        )

        def new_scorer(estimator, X, y) -> dict:
            scores = scorers(estimator, X, y)
            if isinstance(scores, dict):
                return internal_scorer(estimator, X, y) | scores
            return internal_scorer(estimator, X, y) | {"score": scores}

        new_scorers = new_scorer

        # In this specific case, we can't know if there is overlap between the
        # user-defined scores and ours, so we take the least risky option
        # which is to say we added nothing; that way, we won't remove anything
        # after cross-validation is computed
        added_scorers = []

    return new_scorers, added_scorers


def _strip_cv_results_scores(cv_results: dict, added_scorers: list[str]) -> dict:
    """Remove information about `added_scorers` in `cv_results`.

    Parameters
    ----------
    cv_results : dict
        A dict of the form returned by scikit-learn's cross_validate function.
    added_scorers : list[str]
        A list of scorers in `cv_results` which should be removed.

    Returns
    -------
    dict
        A new cv_results dict, with the specified scorers information removed.
    """
    # Takes care both of train and test scores
    return {
        k: v
        for k, v in cv_results.items()
        if not any(added_scorer in k for added_scorer in added_scorers)
    }


def cross_validate(*args, project: Optional[Project] = None, **kwargs) -> dict:
    """Evaluate estimator by cross-validation and output UI-friendly object.

    This function wraps scikit-learn's :func:`~sklearn.model_selection.cross_validate`
    function, to provide more context and facilitate the analysis.
    As such, the arguments are the same as scikit-learn's ``cross_validate`` function.

    The dict returned by this function is a strict super-set of the one returned by
    scikit-learn's :func:`~sklearn.model_selection.cross_validate`.

    For a user guide and in-depth example, see :ref:`example_cross_validate`.

    Parameters
    ----------
    *args
        Positional arguments accepted by scikit-learn's
        :func:`~sklearn.model_selection.cross_validate`,
        such as ``estimator`` and ``X``.
    project : Project, optional
        A project to save cross-validation data into. If None, no save is performed.
    **kwargs
        Additional keyword arguments accepted by scikit-learn's
        :func:`~sklearn.model_selection.cross_validate`.

    Returns
    -------
    cv_results : dict
        A dict of the form returned by scikit-learn's
        :func:`~sklearn.model_selection.cross_validate` function.

    Examples
    --------
    >>> def prepare_cv():
    ...     from sklearn import datasets, linear_model
    ...     diabetes = datasets.load_diabetes()
    ...     X = diabetes.data[:150]
    ...     y = diabetes.target[:150]
    ...     lasso = linear_model.Lasso()
    ...     return lasso, X, y

    >>> project = skore.load("project.skore")  # doctest: +SKIP
    >>> lasso, X, y = prepare_cv()  # doctest: +SKIP
    >>> cross_validate(lasso, X, y, cv=3, project=project)  # doctest: +SKIP
    {'fit_time': array(...), 'score_time': array(...), 'test_score': array(...)}
    """
    import sklearn.model_selection

    # Recover specific arguments
    estimator = args[0] if len(args) >= 1 else kwargs.get("estimator")
    X = args[1] if len(args) >= 2 else kwargs.get("X")
    y = args[2] if len(args) == 3 else kwargs.get("y")

    try:
        scorers = kwargs.pop("scoring")
    except KeyError:
        scorers = None

    # Extend scorers with other relevant scorers
    scorers_to_add = _get_scorers_to_add(estimator, y)
    new_scorers, added_scorers = _add_scorers(scorers, scorers_to_add)

    cv_results = sklearn.model_selection.cross_validate(
        *args, **kwargs, scoring=new_scorers
    )

    cross_validation_item = CrossValidationItem.factory(cv_results, estimator, X, y)

    if project is not None:
        try:
            cv_results_history = project.get_item_versions("cross_validation")
        except KeyError:
            cv_results_history = []

        agg_cross_validation_item = CrossValidationAggregationItem.factory(
            cv_results_history + [cross_validation_item]
        )

        project.put_item("cross_validation_aggregated", agg_cross_validation_item)
        project.put_item("cross_validation", cross_validation_item)

    # If in a IPython context (e.g. Jupyter notebook), display the plot
    with contextlib.suppress(ImportError):
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.display import display

        if InteractiveShell.initialized():
            display(cross_validation_item.plot)

    # Remove information related to our scorers, so that our return value is
    # the same as sklearn's
    stripped_cv_results = _strip_cv_results_scores(cv_results, added_scorers)

    # Add explicit metric to result (rather than just "test_score")
    if isinstance(scorers, str):
        if kwargs.get("return_train_score") is not None:
            stripped_cv_results[f"train_{scorers}"] = stripped_cv_results["train_score"]
        stripped_cv_results[f"test_{scorers}"] = stripped_cv_results["test_score"]

    return stripped_cv_results


def register_accessor(name, target_cls):
    """Register an accessor for a class.

    Parameters
    ----------
    name : str
        The name of the accessor.
    target_cls : type
        The class to register the accessor for.
    """

    def decorator(accessor_cls):
        def getter(self):
            attr = f"_accessor_{accessor_cls.__name__}"
            if not hasattr(self, attr):
                setattr(self, attr, accessor_cls(self))
            return getattr(self, attr)

        setattr(target_cls, name, property(getter))
        return accessor_cls

    return decorator


class CrossValidationReporter:
    """Analyse the output of scikit-learn's cross_validate function.

    Parameters
    ----------
    cv_results : dict
        A dict of the form returned by scikit-learn's
        :func:`~sklearn.model_selection.cross_validate` function.
    data : {array-like, sparse matrix}
        The data used to fit the estimator.
    target : array-like
        The target vector.
    sample_weight : array-like, default=None
        The sample weights.
    """

    def __init__(self, cv_results, data, target, sample_weight=None):
        required_keys = {"estimator": "return_estimator", "indices": "return_indices"}
        missing_keys = [key for key in required_keys if key not in cv_results]

        if missing_keys:
            missing_params = [f"{required_keys[key]}=True" for key in missing_keys]
            raise RuntimeError(
                f"The keys {missing_keys} are required in `cv_results` to create a "
                f"`MetricsReporter` instance, but they are not found. You need "
                f"to set {', '.join(missing_params)} in "
                "`sklearn.model_selection.cross_validate()`."
            )

        self.cv_results = cv_results
        self.data = data
        self.target = target
        self.sample_weight = sample_weight

        self._rng = np.random.default_rng(time.time_ns())
        # It could be included in cv_results but let's avoid to mutate it for the moment
        self._hash = [
            self._rng.integers(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max)
            for _ in range(len(cv_results["estimator"]))
        ]
        self._cache = {}
        self._ml_task = _find_ml_task(cv_results["estimator"][0], target)

    def _get_cached_response_values(
        self,
        *,
        hash,
        estimator,
        X,
        response_method,
        pos_label=None,
    ):
        prediction_method = _check_response_method(estimator, response_method).__name__
        if prediction_method in ("predict_proba", "decision_function"):
            # pos_label is only important in classification and with probabilities
            # and decision functions
            cache_key = (hash, pos_label, prediction_method)
        else:
            cache_key = (hash, prediction_method)

        if cache_key in self._cache:
            return self._cache[cache_key]

        predictions, _ = _get_response_values(
            estimator,
            X=X,
            response_method=prediction_method,
            pos_label=pos_label,
            return_response_method_used=False,
        )
        self._cache[cache_key] = predictions

        return predictions


def _check_supported_ml_task(supported_ml_tasks):
    def check(accessor):
        supported_task = any(
            task in accessor._parent._ml_task for task in supported_ml_tasks
        )

        if not supported_task:
            raise AttributeError(
                f"The {accessor._parent._ml_task} task is not a supported task by "
                f"function called. The supported tasks are {supported_ml_tasks}."
            )

        return True

    return check


@register_accessor("plot", CrossValidationReporter)
class _PlotAccessor:
    def __init__(self, parent):
        self._parent = parent

    @available_if(
        _check_supported_ml_task(supported_ml_tasks=["binary-classification"])
    )
    def roc(
        self,
        positive_class=None,
        name=None,
        plot_chance_level=True,
        chance_level_kw=None,
        despine=True,
        backend="matplotlib",
    ):
        """Plot the ROC curve.

        Parameters
        ----------
        positive_class : str, default=None
            The positive class.
        name : str, default=None
            The name of the plot.
        plot_chance_level : bool, default=True
            Whether to plot the chance level.
        chance_level_kw : dict, default=None
            The keyword arguments for the chance level.
        despine : bool, default=True
            Whether to despine the plot. Only relevant for matplotlib backend.
        backend : {"matplotlib", "plotly"}, default="matplotlib"
            The backend to use for plotting.

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
            The ROC curve plot.
        """
        prediction_method = ["predict_proba", "decision_function"]

        ax = None
        for fold_idx, (hash, estimator, test_indices) in enumerate(
            zip(
                self._parent._hash,
                self._parent.cv_results["estimator"],
                self._parent.cv_results["indices"]["test"],
            )
        ):
            y_pred = self._parent._get_cached_response_values(
                hash=hash,
                estimator=estimator,
                X=self._parent.data,
                response_method=prediction_method,
                pos_label=positive_class,
            )

            y_true_split = _safe_indexing(self._parent.target, test_indices)
            y_pred_split = _safe_indexing(y_pred, test_indices)
            if self._parent.sample_weight is not None:
                sample_weight_split = _safe_indexing(
                    self._parent.sample_weight, test_indices
                )
            else:
                sample_weight_split = None

            cache_key = (hash, RocCurveDisplay.__name__)

            # trick to have the chance level only in the last plot
            if fold_idx == len(self._parent._hash) - 1:
                plot_chance_level_ = plot_chance_level
            else:
                plot_chance_level_ = False

            if name is None:
                name_ = f"{estimator.__class__.__name__} - Fold {fold_idx}"
            else:
                name_ = f"{name} - Fold {fold_idx}"

            if cache_key in self._parent._cache:
                display = self._parent._cache[cache_key].plot(
                    ax=ax,
                    backend=backend,
                    name=name_,
                    plot_chance_level=plot_chance_level_,
                    chance_level_kw=chance_level_kw,
                    despine=despine,
                )
            else:
                display = RocCurveDisplay.from_predictions(
                    y_true_split,
                    y_pred_split,
                    sample_weight=sample_weight_split,
                    pos_label=positive_class,
                    ax=ax,
                    backend=backend,
                    name=name_,
                    plot_chance_level=plot_chance_level_,
                    chance_level_kw=chance_level_kw,
                    despine=despine,
                )
                self._parent._cache[cache_key] = display

            # overwrite for the subsequent plots
            ax = display.ax_

        return display.figure_


@register_accessor("metrics", CrossValidationReporter)
class _MetricsAccessor:
    def __init__(self, parent):
        self._parent = parent

    # TODO: should build on the `add_scorers` function
    def report_stats(self, scoring=None, positive_class=1):
        """Report statistics for the metrics.

        Parameters
        ----------
        scoring : list of str, default=None
            The metrics to report.
        positive_class : int, default=1
            The positive class.

        Returns
        -------
        pd.DataFrame
            The statistics for the metrics.
        """
        if scoring is None:
            scoring = ["accuracy", "precision", "recall"]

        scores = []

        for metric in scoring:
            metric_fn = getattr(self, metric)
            import inspect

            if "positive_class" in inspect.signature(metric_fn).parameters:
                scores.append(metric_fn(positive_class=positive_class))
            else:
                scores.append(metric_fn())

        has_multilevel = any(
            isinstance(score, pd.DataFrame) and isinstance(score.columns, pd.MultiIndex)
            for score in scores
        )

        if has_multilevel:
            # Convert single-level dataframes to multi-level
            for i, score in enumerate(scores):
                if hasattr(score, "columns") and not isinstance(
                    score.columns, pd.MultiIndex
                ):
                    scores[i].columns = pd.MultiIndex.from_tuples(
                        [(col, "") for col in score.columns]
                    )

        return pd.concat(scores, axis=1)

    def _compute_metric_scores(
        self,
        metric_fn,
        *,
        response_method,
        pos_label=None,
        metric_name=None,
        **metric_kwargs,
    ):
        scores = []

        for hash, estimator, test_indices in zip(
            self._parent._hash,
            self._parent.cv_results["estimator"],
            self._parent.cv_results["indices"]["test"],
        ):
            y_pred = self._parent._get_cached_response_values(
                hash=hash,
                estimator=estimator,
                X=self._parent.data,
                response_method=response_method,
                pos_label=pos_label,
            )

            cache_key = (hash, metric_name)
            metric_params = inspect.signature(metric_fn).parameters
            if "pos_label" in metric_params:
                cache_key += (pos_label,)
            if "average" in metric_params:
                cache_key += (metric_kwargs["average"],)

            if cache_key in self._parent._cache:
                score = self._parent._cache[cache_key]
            else:
                y_true_split = _safe_indexing(self._parent.target, test_indices)
                y_pred_split = _safe_indexing(y_pred, test_indices)

                sample_weight_split = None
                if self._parent.sample_weight is not None:
                    sample_weight_split = _safe_indexing(
                        self._parent.sample_weight, test_indices
                    )

                metric_params = inspect.signature(metric_fn).parameters
                kwargs = {**metric_kwargs}
                if "pos_label" in metric_params:
                    kwargs.update(pos_label=pos_label)

                score = metric_fn(
                    y_true_split,
                    y_pred_split,
                    sample_weight=sample_weight_split,
                    **kwargs,
                )

                self._parent._cache[cache_key] = score

            scores.append(score)
        scores = np.array(scores)

        metric_name = metric_name or metric_fn.__name__

        if self._parent._ml_task in [
            "binary-classification",
            "multiclass-classification",
        ]:
            if scores.ndim == 1:
                columns = [metric_name]
            else:
                classes = self._parent.cv_results["estimator"][0].classes_
                columns = [[metric_name] * len(classes), classes]
        elif self._parent._ml_task == "regression":
            if scores.ndim == 1:
                columns = [metric_name]
            else:
                columns = [
                    [metric_name] * scores.shape[1],
                    [f"Output #{i}" for i in range(scores.shape[1])],
                ]
        else:
            columns = None
        return pd.DataFrame(scores, columns=columns)

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def accuracy(self):
        """Compute the accuracy score.

        Returns
        -------
        pd.DataFrame
            The accuracy score.
        """
        return self._compute_metric_scores(
            metrics.accuracy_score, response_method="predict", metric_name="accuracy"
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision(self, average="binary", positive_class=1):
        """Compute the precision score.

        Parameters
        ----------
        average : {"binary", "micro", "macro", "weighted", "samples"}, default="binary"
            The average to compute the precision score.
        positive_class : int, default=1
            The positive class.

        Returns
        -------
        pd.DataFrame
            The precision score.
        """
        return self._compute_metric_scores(
            metrics.precision_score,
            response_method="predict",
            pos_label=positive_class,
            metric_name="precision",
            average=average,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def recall(self, average="binary", positive_class=1):
        """Compute the recall score.

        Parameters
        ----------
        average : {"binary", "micro", "macro", "weighted", "samples"}, default="binary"
            The average to compute the recall score.
        positive_class : int, default=1
            The positive class.

        Returns
        -------
        pd.DataFrame
            The recall score.
        """
        return self._compute_metric_scores(
            metrics.recall_score,
            response_method="predict",
            pos_label=positive_class,
            metric_name="recall",
            average=average,
        )

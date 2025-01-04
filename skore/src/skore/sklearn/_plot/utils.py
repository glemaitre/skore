import inspect
from io import StringIO

from rich.console import Console
from rich.tree import Tree
from sklearn.utils._plotting import check_consistent_length, check_matplotlib_support
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_pos_label_consistency


class HelpDisplayMixin:
    """Mixin class to add help functionality to a class."""

    def _get_attributes_for_help(self):
        """Get the attributes ending with '_' to display in help."""
        attributes = []
        for name in dir(self):
            if name.endswith("_") and not name.startswith("_"):
                attributes.append(name)
        return sorted(attributes)

    def _get_methods_for_help(self):
        """Get the public methods to display in help."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        filtered_methods = []
        for name, method in methods:
            is_private = name.startswith("_")
            is_class_method = inspect.ismethod(method) and method.__self__ is type(self)
            is_help_method = name == "help"
            if not (is_private or is_class_method or is_help_method):
                filtered_methods.append((name, method))
        return sorted(filtered_methods)

    def _create_help_tree(self):
        """Create a rich Tree with attributes and methods."""
        tree = Tree(f"📊 {self.__class__.__name__}")

        attributes = self._get_attributes_for_help()
        if attributes:
            attr_branch = tree.add("📝 Attributes to tweak your plot")
            # Ensure figure_ and ax_ are first
            sorted_attrs = sorted(attributes)
            if "ax_" in sorted_attrs:
                sorted_attrs.remove("ax_")
            if "figure_" in sorted_attrs:
                sorted_attrs.remove("figure_")
            sorted_attrs = ["figure_", "ax_"] + [
                attr for attr in sorted_attrs if attr not in ["figure_", "ax_"]
            ]
            for attr in sorted_attrs:
                attr_branch.add(attr)

        methods = self._get_methods_for_help()
        if methods:
            method_branch = tree.add("🔧 Methods")
            for name, method in methods:
                description = (
                    method.__doc__.split("\n")[0]
                    if method.__doc__
                    else "No description available"
                )
                method_branch.add(f"{name} - {description}")

        return tree

    def help(self):
        """Display available attributes and methods using rich."""
        from skore import console  # avoid circular import

        console.print(self._create_help_tree())

    def __repr__(self):
        """Return a string representation using rich."""
        console = Console(file=StringIO(), force_terminal=False)
        console.print(self._create_help_tree())
        return console.file.getvalue()


class _ClassifierCurveDisplayMixin:
    """Mixin class to be used in Displays requiring a binary classifier.

    The aim of this class is to centralize some validations regarding the estimator and
    the target and gather the response of the estimator.
    """

    def _validate_plot_params(self, *, ax=None, name=None):
        check_matplotlib_support(f"{self.__class__.__name__}.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        name = self.estimator_name if name is None else name
        return ax, ax.figure, name

    @classmethod
    def _validate_from_predictions_params(
        cls, y_true, y_pred, *, sample_weight=None, pos_label=None, name=None
    ):
        check_matplotlib_support(f"{cls.__name__}.from_predictions")

        target_type = type_of_target(y_true)
        if target_type not in ("binary", "multiclass"):
            raise ValueError(
                f"The target y is not binary. Got {target_type} type of target."
            )

        check_consistent_length(y_true, y_pred, sample_weight)

        if target_type == "binary":
            pos_label = _check_pos_label_consistency(pos_label, y_true)

        name = name if name is not None else "Classifier"

        return pos_label, name


def _despine_matplotlib_axis(ax):
    """Despine the matplotlib axis."""
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_bounds(0, 1)


def _validate_style_kwargs(default_style_kwargs, user_style_kwargs):
    """Create valid style kwargs by avoiding Matplotlib alias errors.

    Matplotlib raises an error when, for example, 'color' and 'c', or 'linestyle' and
    'ls', are specified together. To avoid this, we automatically keep only the one
    specified by the user and raise an error if the user specifies both.

    Parameters
    ----------
    default_style_kwargs : dict
        The Matplotlib style kwargs used by default in the scikit-learn display.
    user_style_kwargs : dict
        The user-defined Matplotlib style kwargs.

    Returns
    -------
    valid_style_kwargs : dict
        The validated style kwargs taking into account both default and user-defined
        Matplotlib style kwargs.
    """
    invalid_to_valid_kw = {
        "ls": "linestyle",
        "c": "color",
        "ec": "edgecolor",
        "fc": "facecolor",
        "lw": "linewidth",
        "mec": "markeredgecolor",
        "mfcalt": "markerfacecoloralt",
        "ms": "markersize",
        "mew": "markeredgewidth",
        "mfc": "markerfacecolor",
        "aa": "antialiased",
        "ds": "drawstyle",
        "font": "fontproperties",
        "family": "fontfamily",
        "name": "fontname",
        "size": "fontsize",
        "stretch": "fontstretch",
        "style": "fontstyle",
        "variant": "fontvariant",
        "weight": "fontweight",
        "ha": "horizontalalignment",
        "va": "verticalalignment",
        "ma": "multialignment",
    }
    for invalid_key, valid_key in invalid_to_valid_kw.items():
        if invalid_key in user_style_kwargs and valid_key in user_style_kwargs:
            raise TypeError(
                f"Got both {invalid_key} and {valid_key}, which are aliases of one "
                "another"
            )
    valid_style_kwargs = default_style_kwargs.copy()

    for key in user_style_kwargs:
        if key in invalid_to_valid_kw:
            valid_style_kwargs[invalid_to_valid_kw[key]] = user_style_kwargs[key]
        else:
            valid_style_kwargs[key] = user_style_kwargs[key]

    return valid_style_kwargs

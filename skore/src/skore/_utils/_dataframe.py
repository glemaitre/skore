from typing import TYPE_CHECKING, Any, TypeAlias, cast

import narwhals as nw
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import ArrayLike

from skore._externals._sklearn_compat import check_array

if TYPE_CHECKING:
    import polars as pl

    UserDataFrame: TypeAlias = pd.DataFrame | pl.DataFrame
    UserSeries: TypeAlias = pd.Series | pl.Series
else:
    UserDataFrame: TypeAlias = pd.DataFrame
    UserSeries: TypeAlias = pd.Series

UserTarget: TypeAlias = UserSeries | UserDataFrame


def _ensure_string_column_names(df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
    if all(isinstance(col, str) for col in df.columns):
        return df
    return df.rename({col: str(col) for col in df.columns})


def _eager_backend(obj: ArrayLike) -> nw.Implementation | None:
    """Return the dataframe backend of ``obj``, or None when it is not tabular."""
    if nw.dependencies.is_into_series(obj) or nw.dependencies.is_into_dataframe(obj):
        return nw.from_native(obj, allow_series=True).implementation
    return None


def _frame_from_numpy(
    array: np.ndarray, columns: list[str], backend: nw.Implementation | None
) -> UserDataFrame:
    if backend is not nw.Implementation.POLARS:
        return pd.DataFrame(array, columns=columns)
    return cast(
        "UserDataFrame",
        nw.from_numpy(array, schema=columns, backend=backend).to_native(),
    )


def _normalize_X_as_dataframe(
    X: ArrayLike, *, backend: nw.Implementation | None = None
) -> UserDataFrame:
    """Normalize feature data as a DataFrame with string column names."""
    if sp.issparse(X):
        raise NotImplementedError(
            "Data analysis via skrub is currently not supported for sparse matrices. "
            "Please use dense data."
        )

    if not nw.dependencies.is_into_dataframe(X):
        X = check_array(
            X,
            accept_sparse=False,
            ensure_2d=True,
            ensure_all_finite=False,
            dtype=None,
        )
        X = cast(np.ndarray, X)
        columns = [f"Feature {i}" for i in range(X.shape[1])]
        return _frame_from_numpy(X, columns, backend)

    return _ensure_string_column_names(nw.from_native(X)).to_native()


def _normalize_y_as_dataframe(
    y: ArrayLike, *, backend: nw.Implementation | None = None
) -> UserDataFrame:
    """Normalize target data as a DataFrame with predictable column names."""
    if sp.issparse(y):
        raise NotImplementedError(
            "Data analysis via skrub is currently not supported for sparse matrices. "
            "Please use dense data."
        )

    if nw.dependencies.is_into_series(y):
        if nw.dependencies.is_polars_series(y):
            series = nw.from_native(y, series_only=True)
            name = series.name if series.name else "Target"
            if not series.name:
                series = series.rename(name)
            return series.to_frame().to_native()

        y_series = cast(pd.Series, y)
        name = y_series.name if y_series.name is not None else "Target"
        return y_series.to_frame(name=name)

    if nw.dependencies.is_into_dataframe(y):
        if nw.dependencies.is_polars_dataframe(y):
            return y

        y_df = cast(pd.DataFrame, y)
        if all(isinstance(col, str) for col in y_df.columns):
            return y_df

        y_df = y_df.copy(deep=False)
        if y_df.shape[1] == 1 and list(y_df.columns) == [0]:
            y_df.columns = ["Target"]
        else:
            y_df.columns = [str(col) for col in y_df.columns]
        return y_df

    y = np.asarray(y)

    columns = ["Target"] if y.ndim == 1 else [f"Target {i}" for i in range(y.shape[1])]
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return _frame_from_numpy(y, columns, backend)


def _deduplicate_target_columns(
    y: nw.DataFrame[Any], taken: list[str]
) -> nw.DataFrame[Any]:
    """Rename target columns clashing with feature names, keeping names unique."""
    renaming, seen = {}, list(taken)
    for column in y.columns:
        if column not in seen:
            seen.append(column)
            continue
        candidate = f"{column}_target"
        suffix = 1
        while candidate in seen:
            candidate = f"{column}_target_{suffix}"
            suffix += 1
        renaming[column] = candidate
        seen.append(candidate)
    return y.rename(renaming) if renaming else y


def _combine_X_y(
    X_frame: nw.DataFrame[Any], y_frame: nw.DataFrame[Any]
) -> nw.DataFrame[Any]:
    """Concatenate features and target horizontally, row by row.

    Both frames are brought to a common backend, target columns clashing with
    feature names are renamed, and rows are aligned by position.
    """
    if y_frame.implementation is not X_frame.implementation:
        if X_frame.implementation is nw.Implementation.POLARS:
            y_frame = nw.from_native(y_frame.to_polars())
        else:
            # pandas -> polars needs no pyarrow, unlike the other direction
            X_frame = nw.from_native(X_frame.to_polars())
            y_frame = nw.from_native(y_frame.to_polars())

    y_frame = _deduplicate_target_columns(y_frame, X_frame.columns)

    if X_frame.implementation.is_pandas_like():
        # pandas concatenates on index labels, so give y the row labels of X
        y_native = cast(pd.DataFrame, y_frame.to_native())
        X_native = cast(pd.DataFrame, X_frame.to_native())
        y_frame = nw.from_native(y_native.set_axis(X_native.index))

    return nw.concat([X_frame, y_frame], how="horizontal")


def _concat_X_y(X: ArrayLike, y: ArrayLike) -> UserDataFrame:
    """Combine raw features and target into a single DataFrame."""
    backend = _eager_backend(X) or _eager_backend(y)
    return _combine_X_y(
        nw.from_native(_normalize_X_as_dataframe(X, backend=backend)),
        nw.from_native(_normalize_y_as_dataframe(y, backend=backend)),
    ).to_native()


def _concat_vertical(
    a: ArrayLike, b: ArrayLike
) -> UserDataFrame | UserSeries | np.ndarray:
    """Concatenate two tabular objects vertically, preserving the native backend."""
    if nw.dependencies.is_into_series(a):
        combined = nw.concat(
            [
                nw.from_native(a, allow_series=True, series_only=True).to_frame(),
                nw.from_native(b, allow_series=True, series_only=True).to_frame(),
            ],
            how="vertical",
        )
        return combined.get_column(combined.columns[0]).to_native()

    if nw.dependencies.is_into_dataframe(a):
        return nw.concat(
            [nw.from_native(a), nw.from_native(b)],
            how="vertical",
        ).to_native()

    return np.concatenate([np.asarray(a), np.asarray(b)])

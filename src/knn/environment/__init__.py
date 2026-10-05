"""Environment estimators.

Each estimator is a callable registered under a name with the same signature:

    estimator(tracer_xyz, target_xyz, k, self_index=None) -> dict

The dict maps a quantity name ('r', 'rho', 'log1p_delta', ...) to an array
with one value per target. The build step names the columns
'<quantity>_<name>_k<k>'.
"""

ESTIMATORS = {}


def register(name):
    """Decorator registering an estimator under 'name'."""
    def decorator(func):
        if name in ESTIMATORS:
            raise ValueError(f"estimator '{name}' is already registered")
        ESTIMATORS[name] = func
        return func
    return decorator


def get_estimator(name):
    """Look up a registered estimator, with a clear error for unknown names."""
    try:
        return ESTIMATORS[name]
    except KeyError:
        raise KeyError(
            f"unknown environment estimator '{name}'; available: {sorted(ESTIMATORS)}"
        ) from None


from . import knn3d  # noqa: E402,F401  (registers 'knn3d')

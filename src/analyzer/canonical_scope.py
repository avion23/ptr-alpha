"""Keep every canonical read in a dated operation on the same decision day."""
from contextlib import nullcontext
from functools import wraps
from inspect import signature
from datetime import timedelta
import pandas as pd


def decision_history(db, start, decisions, *, lookback_days=28):
    frames = []
    for decision in decisions:
        scope_date = getattr(db, "_canonical_as_of", None) or decision
        scope = db.canonical_as_of(scope_date) if hasattr(db, "canonical_as_of") else nullcontext()
        with scope:
            frames.append(db.get_transactions_by_date_range(
                max(pd.Timestamp(start), pd.Timestamp(decision) - timedelta(days=lookback_days)), decision
            ))
    return frames


def canonical_decision(date_parameter):
    def decorate(function):
        parameters = signature(function)

        @wraps(function)
        def scoped(*args, **kwargs):
            bound = parameters.bind(*args, **kwargs)
            decision = bound.arguments.get(date_parameter)
            source = next(iter(bound.arguments.values()))
            db = getattr(source, "db", source)
            # Preserve an enclosing replay frame (e.g. disclosure-frame
            # capture) instead of narrowing nested holdings reads back to the
            # decision date. Outside an explicit frame, use the decision day.
            scope_date = getattr(db, "_canonical_as_of", None) or decision
            scope = (db.canonical_as_of(scope_date)
                     if scope_date is not None and hasattr(db, "canonical_as_of")
                     else nullcontext())
            with scope:
                return function(*args, **kwargs)
        return scoped
    return decorate

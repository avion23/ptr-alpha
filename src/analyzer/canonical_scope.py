"""Keep every canonical read in a dated operation on the same decision day."""
from contextlib import nullcontext
from functools import wraps
from inspect import signature


def decision_history(db, start, decisions):
    frames = []
    for decision in decisions:
        scope = db.canonical_as_of(decision) if hasattr(db, "canonical_as_of") else nullcontext()
        with scope:
            frames.append(db.get_transactions_by_date_range(start, decision))
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
            scope = (db.canonical_as_of(decision)
                     if decision is not None and hasattr(db, "canonical_as_of")
                     else nullcontext())
            with scope:
                return function(*args, **kwargs)
        return scoped
    return decorate

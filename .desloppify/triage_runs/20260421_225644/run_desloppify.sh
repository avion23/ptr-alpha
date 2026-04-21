#!/bin/sh
export PYTHONPATH=/Users/avion/.local/pipx/venvs/desloppify/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}
exec /Users/avion/.local/pipx/venvs/desloppify/bin/python -m desloppify.cli "$@"

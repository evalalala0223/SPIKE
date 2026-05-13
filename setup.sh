#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m pip install -r "$ROOT_DIR/requirements.txt"
python -m pip install -e "$ROOT_DIR/agent"

export PYTHONPATH="$PYTHONPATH:$ROOT_DIR"
export PYTHONPATH="$PYTHONPATH:$ROOT_DIR/env"
export PYTHONPATH="$PYTHONPATH:$ROOT_DIR/agent"

echo "StarDojo environment is ready for this shell."
echo "PYTHONPATH=$PYTHONPATH"

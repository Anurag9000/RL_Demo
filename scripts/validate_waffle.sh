#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m compileall -q arguments.py train.py waffle_pi crowd_nav crowd_sim rl gst_updated
python -m unittest discover -s tests -v

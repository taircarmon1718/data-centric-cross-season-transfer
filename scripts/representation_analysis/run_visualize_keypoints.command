#!/bin/bash
# Double-click this file in Finder to run the keypoint visualizer with the project's Python
PYTHON_EXEC="/opt/anaconda3/envs/env-for-ml/bin/python"
SCRIPT="$(dirname "$0")/visualize_keypoints.py"
"$PYTHON_EXEC" "$SCRIPT" &>/dev/null &
exit 0


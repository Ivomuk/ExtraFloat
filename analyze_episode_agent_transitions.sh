#!/usr/bin/env bash
# Linux equivalent of analyze_episode_agent_transitions.bat -- same behavior.
set -u

python3 scripts/analyze_episode_agent_transitions.py "$@"

status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "ERROR: analyze_episode_agent_transitions.py failed with exit code $status"
    exit "$status"
fi

@echo off
python calibrate_scorecard.py ^
    --agents data/agent_profile_snapshot.csv ^
    --out scorecards/capacity_scorecard_v1.json ^
    --force

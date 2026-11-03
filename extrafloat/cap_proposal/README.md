# Burden limit design (proposal, not live)

A proposed replacement for the engine's capacity ceiling and experience ramp:
the limit is capped at **a multiple of the agent's monthly commission**
(6-month commission / 6), and the multiple depends on repayment experience and current arrears.
Any increase beyond what the current 1,000,000 UGX engine would give is limited to
**1.25 × the agent's largest past loan**. See the docstring at the top of `limit_design.py`.

### Agent-category guard (`agent_tier_guard`)

Agent categories come from the 6-month commission, and each category has a maximum limit:

| Category | Below Threshold | New Bronze | Bronze | Silver | Gold | Platinum | Titanium | Diamond |
|---|---|---|---|---|---|---|---|---|
| Maximum (UGX) | 0 | 50,000 | 100,000 | 250,000 | 350,000 | 500,000 | 750,000 | 1,000,000 |

The design overwrites the engine's tier ceiling with 4 × monthly commission. Without a guard, that breaks
the category maximums:
- Below Threshold agents would get a limit.
- New Bronze could reach about 67k and Bronze about 167k.
- Diamond could reach the 5M engine ceiling.

`agent_tier_guard` re-applies the category maximum after the design. It only ever lowers a limit.

| Setting | Effect |
|---|---|
| `enforce` (default) | Every agent is capped at its category maximum; Below Threshold agents get 0. Nobody goes above 1,000,000. |
| `enforce_below_diamond` | Same as `enforce`, except Diamond agents can go above 1,000,000, limited by the governor and the 5M ceiling. |
| `off` | The design as originally written, with no category guard. |

Compare all three in one run with `--agent-tier-guard enforce off enforce_below_diamond`. Besides one folder per setting, this writes side-by-side files to `<out-dir>`:
- `<settings>__guard_comparison_agents.csv`: one row per agent, with the columns `limit_current`, `limit_category_enforced`, `limit_category_not_enforced` and `limit_enforce_below_diamond`
- `<settings>__guard_comparison_by_category.csv` and `.md`: totals and maximums per category for each setting, plus an ALL AGENTS row Every `summary.md`
has a by-category table: each category's maximum, how many proposed limits sit above it, and how many limits
the guard capped. The audit columns are `design_agent_category`, `design_agent_tier_max`,
`design_limit_before_tier_guard` and `design_agent_tier_guard_binding`.

**Nothing here changes the live engine.** `limit_design.py` swaps two engine functions in only for the
length of its own call and puts them back afterwards. The engine source (`extrafloat/engine/`) and
`run_engine.py` are not modified.

## Files

| File | What it is |
|---|---|
| `limit_design.py` | The design itself (`run_burden_design`) |
| `limit_design_proposal.json` | The proposed settings, pending Credit Risk sign-off |
| `limit_design_*candidate*.json` | Alternative settings (s4–s6, 2m lifetime) |
| `run_engine_wrapper.py` | The original local pipeline runner (Windows `F:\` paths, PD join, coverage fix) |
| `run_burden_trial.py` | **Portable side-by-side trial**: live engine vs the design, on the same agents |
| `synthetic_inputs.py` | Synthetic extracts for a dry run of the trial |

## Running the trial

```bash
# Real warehouse extracts
python -m extrafloat.cap_proposal.run_burden_trial \
    --transaction  data/agent_profile_snapshot.csv \
    --borrower     data/borrower_history.csv \
    --loan-summary data/loan_summary.csv \
    --loan-history data/loan_history_snapshot.csv \
    --all-candidates \
    --out-dir      output/burden_trial

# Dry run on synthetic agents (checks the plumbing and the rules, not the numbers)
python -m extrafloat.cap_proposal.run_burden_trial --synthetic 5000 --all-candidates --out-dir output/burden_trial_synth
```

- **Current limit** = `run_extrafloat_limit_engine()` exactly as `run_engine.py` runs it in production
  (no loan history, no PD). Pass `--current-with-loan-history` to give the current run the same loan
  history as the design, which turns on the engine's unresolved-loan haircut.
- **Proposed limit** = `run_burden_design()` with each settings file (`--design-config a.json b.json`,
  or `--all-candidates`).
- **Outputs**, per settings file in `<out-dir>/<settings>/`:
  - `agents.csv`: one row per agent with the current limit, the proposed limit, the stage, the multiple
    applied and the binding constraint
  - `summary.md` and `summary.json`: totals, direction of change, and tables by stage and by binding constraint
  - `<out-dir>/index.csv`: one line per settings file

### Inputs the design needs

| Column | Source |
|---|---|
| `commission` (6-month sum) | transaction / agent profile extract |
| `closed_loan_count`, `consecutive_on_time_loans`, `active_loan_days_aging_at_snapshot`, `anomaly_open_at_snapshot` | loan history snapshot (`--loan-history`), required |
| `lifetime_worst_days_to_principal_cure` | derived by the trial as `lifetime_worst_hours_to_principal_cure / 24` |
| `lifetime_on_time_rate` (only when `clean_on_time_source = "lifetime"`) | derived by the trial from `lifetime_on_time_24h_rate` |
| `current_dpd_at_snapshot`, `written_off_at_snapshot`, `worst_dpd_material_any_loan_at_snapshot`, `days_since_last_repayment_any_loan_at_snapshot` (s4–s6 candidates only) | the **newer** loan history extract. These are not in `data/loan_history_snapshot_query.txt` |

If an input is missing, the trial stops with an error for that settings file and moves on to the next one.
It never fills a missing input silently.

### Differences from `run_engine_wrapper.py`

The trial does not apply the wrapper's local corrections: the PD join (`pd_join`), the coverage fix and
the overdue-only haircut scope. The wrapper's `--capacity-scaling` option has no effect under the burden
design, because the design replaces `capacity_cap` entirely. So the absolute limits can differ from a wrapper
run, but the design logic is the same code.

## Tests

```bash
python -m pytest tests/cap_proposal -q
```

The tests cover:
- the live engine's output is identical before and after a design run;
- the engine functions are restored, even when the design raises an error;
- the ladder stages and multiples;
- the governor bound, the ceiling and the rounding;
- an end-to-end CLI run of every candidate settings file.

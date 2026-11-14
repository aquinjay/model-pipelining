# Configuration

This directory is reserved for future standalone examples. **No current command
loads a configuration file from here.** Choose settings through Python keyword
arguments or the two run scripts; the runner saves them automatically in the run
manifest for reference and reproduction.

| Setting group | How it is selected | Where the run saves it |
|---|---|---|
| Input | `dataset` plus exactly one of `snapshot_id` or `tag`; resolve a tag once | `input` pins the permanent dataset/ID |
| Task | Current daily task plus `minimum_history` (default 30) | `task` |
| Model | `TrainingMean` by default or explicit `LocalLevel(...)` | `model.configuration` and model identity/capabilities |
| Uncertainty request | `interval_levels=(0.8, 0.95)`; both scripts accept `--interval-levels 0.8 0.95` | `evaluation` |
| Local storage | Snapshot `store` and a new `output_directory` | Local locations are separate from permanent input identity |

[The model guide](../docs/local-level.md#settings-and-timing) defines the four
fixed local-level settings and their units. [The uncertainty guide](../docs/uncertainty.md)
explains which declared outputs are requested and how reporting chooses a saved
method/level. The point-only baseline creates no uncertainty just because
interval levels are configured.

Reproduction restores saved input/task/model/evaluation choices with matching
code/environment into a new run directory. There is no automatic tuning or model
registry. Acquisition receipts describe input history and are not configuration
files or permanent input paths. Start with [the walkthrough](../docs/forecasting.md)
for an executable example.

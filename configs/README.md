# Experiment configuration

Reserved for future standalone examples of task specifications, model settings, evaluation protocols, and check plans.

The first baseline uses Python keyword arguments or the options in `scripts/run_baseline.py`; it does not load a configuration file from this directory. `run_backtest` automatically saves the chosen task, model settings, and permanent data reference in the run's JSON manifest. See [the first forecasting workflow](../docs/forecasting.md). A broader configuration format remains open. Examples committed here should be understandable without private data or machine-specific paths.

The current ingestion and snapshot workflow uses Python function calls and needs no configuration file. Call `register_snapshot(receipt.destination, dataset="sales", ingestion_receipt=receipt)` to save the copy's receipt automatically with that registration. See the [complete workflow](../src/model_pipelining/README.md#one-complete-workflow).

The baseline pins the dataset and permanent snapshot ID, resolving any supplied tag once when selecting inputs. Ingestion receipts describe acquisition history; their local paths are not permanent experiment input references.

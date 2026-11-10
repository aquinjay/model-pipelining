# Local run artifacts

Future runs will save evidence here. Git ignores other contents of this directory by default; this orientation file is tracked.

Proposed artifact categories: manifest/configuration, immutable forecasts, scores, check plan/results, simulation references, review/iteration record, and report. Exact formats are undecided. Retention and export for important runs will be designed with the first saved experiment.

There is no model runner yet. The implemented ingestion and snapshot functions save data and registration records under the selected snapshot store, which defaults to `data/versioned/`. When `register_snapshot` receives `ingestion_receipt=receipt`, it automatically saves the receipt with that registration. There is no separate receipt file to put in `runs/`, and registering a snapshot does not start an experiment.

Future runs should save the dataset and permanent snapshot ID used for their inputs. The registration record retains any supplied ingestion source, copy completion time, destination, and byte count separately from registration time. See the [current workflow](../src/model_pipelining/README.md#one-complete-workflow) and [proposed first experiment](../docs/roadmap.md#1-small-synthetic-experiment).

# Experiment configuration

Reserved for small, versioned examples of task specifications, model settings, evaluation protocols, and check plans.

The format is still open. Examples committed here should be understandable without private data or machine-specific paths. Exact record fields will be decided alongside the first runnable slice.

The current ingestion and snapshot workflow uses Python function calls and needs no configuration file. Call `register_snapshot(receipt.destination, dataset="sales", ingestion_receipt=receipt)` to save the copy's receipt automatically with that registration. See the [complete workflow](../src/model_pipelining/README.md#one-complete-workflow).

Future experiment specifications should pin the dataset and permanent snapshot ID, resolving any tag once when selecting inputs. Ingestion receipts describe acquisition history; their local paths are not permanent experiment input references.

# Decisions and open questions

This log distinguishes current direction from proposals. Update it when a choice affects observable behavior or a responsibility boundary.

## Current direction — 2026-11-01

| Decision | Reason |
|---|---|
| Begin with time-indexed numerical forecasting | Gives the first implementation a concrete, bounded task |
| Start on synthetic data with a baseline | Exposes timing and artifact mistakes before model complexity |
| Separate applications, model adapters, and shared execution | Changing model families should not require rebuilding the pipeline |
| Share checking stages/results and let models supply specialized checks | Supports Bayesian workflow and other model families without false equivalences |
| Experiment owns check requirements | Models must not silently bypass required evidence |
| Preserve frozen predictions, provenance, failures, and iteration lineage | Makes comparisons and revisions inspectable |
| Keep documentation independent of the Obsidian vault | A GitHub reader should understand the project without local paths or plugins |

## Working proposals

- Python implementation, starting with a few files in `src/model_pipelining/`.
- Plain local files for configuration and artifacts; select exact formats during the first slice.
- A local-level BDLM with fixed variance settings as the first substantive model.
- One market series as the first real-data application, followed later by a separate operational application.

These are starting choices to revisit deliberately, rather than implemented interfaces or completed experiments.

## Open questions

- What synthetic process best exposes the first failure cases?
- Which Python version, packaging workflow, and minimum dependencies should we use?
- What is the smallest useful request/prediction/check result contract?
- Which checks stop execution, and which allow exploratory evidence collection while blocking release?
- What requires explicit human interpretation in the first review?
- Which artifact formats balance inspectability and reproduction?
- Which instrument/source/target would make the later real-data exercise informative?
- Which GitHub visibility and license should be selected before publication?

## Record a consequential change

Use a dated entry with the question, decision, rationale, alternatives considered, consequences, and evidence or issue/PR links. Preserve superseded choices so later changes remain understandable.

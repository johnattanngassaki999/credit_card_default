# Use of AI tools

This file explains how we used generative AI in this project, what it produced, and what we did ourselves.
We used it as a tutor and a pair programmer. Every piece of code it suggested was read, run and tested by us before it went into the repository.

## Tools

| Tool | Used by | Main use |
|---|---|---|
| Claude (Anthropic), chat interface | Johnattan | Explanations, code suggestions, debugging, review |
| _[tool]_ | _[partner]_ | _[partner to complete]_ |

## What we used AI for (Johnattan: evaluation, models, CLI, tests)

### 1. Understanding the concepts
Before writing code, we asked for explanations of:
- the difference between the ROC curve and the PR curve, and why PR-AUC is our main metric;
- the role of the `C` parameter in logistic regression (regularisation), and the difference between log loss and hinge loss;
- how `class_weight="balanced"` and the decision threshold are two different ways of choosing an operating point.

### 2. Planning
- A day-by-day plan up to the deadline, and a split of the work between the two of us.

### 3. Code
| File | What the AI did | What we did |
|---|---|---|
| `evaluate.py` | Corrected our `compute_metrics`, `get_oof_scores`, `cross_validate_model`, `find_best_threshold` and wrote  the three plot functions; added support for `decision_function` (linear SVM) | Wrote the first versions, chose which metrics to keep, tested every function in the notebook |
| `models.py` | Guided us function by function (names and signatures), then wrote a simplified version of the baseline/pipeline/comparison logic on request | Chose the models (logistic regression, KNN, linear SVM rather than models not seen in class), wrote and debugged `build_pipeline` |
| `config.py`, `config.yaml`, `pyproject.toml` | Drafted the files and the loading function | Adapted the keys to our code (`random_state`, `param_grids`) |
| `features.py` | Resolved a Git merge conflict between our two versions | Checked that the default behaviour did not change |
| `cli.py` | Wrote the command line interface (`train`, `evaluate`, `predict`) and its readable output | Adapted the imports to our `data.py` and `models.py`, ran all three commands |
| `tests/` | Drafted the tests and the `conftest.py` fixture | Fixed import errors, ran `pytest` until all tests passed |
| Notebook | Wrote some cells (comparison tables, PR curve overlay, threshold table, saving the model) and helped us write the comments on the results | Ran the analysis and interpreted the results |

### 4. Debugging and tooling
Virtual environment and `pip install -e` problems, import errors, a YAML `KeyError`, `pytest` fixtures not found, Git merge conflicts, and creating the `.gitignore`.

## Decisions we made ourselves
- Choosing the dataset (UCI Default of Credit Card Clients).
- Using only models seen in class. We considered it more valuable to show that a known model fails on this problem than to use a model we could not explain.
- Choosing logistic regression as the final model, based on our cross-validation results.
- The threshold rule: catching at least 60% of the defaulters, without giving up precision entirely, rather than maximising recall alone.
- Keeping our evaluation separate from the feature and class-weight experiments of the other part.

## Checking the AI's output
The AI made mistakes that we caught or that it corrected when we questioned it. For example:
- it claimed KNN was about five times slower, which our timings did not support;
- it said the tuned threshold would be well below 0.5, which is not true with `class_weight="balanced"`;
- it made a wrong claim about how `find_best_threshold` behaves.

To avoid relying on output we had not checked, we:
- ran every function on the real data and compared it with our expectations;
- kept the test set untouched until the final evaluation;
- wrote and ran unit tests for each module;
- asked for explanations, not only code, so that each of us can explain every part of the project during the defense.

## Partner's part (data, features, EDA, imbalance experiments, interpretation)
_[To be completed by our partner.]_
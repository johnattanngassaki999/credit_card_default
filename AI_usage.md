# Use of AI tools

**Project**: Credit card default prediction (UCI "Default of Credit Card Clients", 30,000 clients, 22% default rate)
**Course**: Machine Learning × Advanced Python Programming, Programme Grande École (2nd year), Aivancity
**Authors**: NGASSAKI Johnattan (models, evaluation, CLI) · TIE Yro (data, features, EDA, experiments)

We both used an AI assistant in this project. This file explains where we used it, what it produced, what we decided ourselves, how we checked its output, and where it was wrong.

## Tools

| Tool | Used by | Period | Main use |
|---|---|---|---|
| Claude (Anthropic), chat interface | Johnattan | October–November 2026 | Explanations, planning, code suggestions, debugging, review |
| Claude (Anthropic), chat interface | Yro | October–November 2026 | Step-by-step code suggestions for data, features, EDA, experiments and tests |

## 1. How we worked with the AI

- **One step at a time.** We did not ask for a finished project. We asked for help with our functions and notebooks, and reported the output back.
- **Course material as the reference.** Both of us chose to use only models seen in class.
- **Each person works on their own part.** Each of us used the AI on their own modules. The only time the AI touched the other person's file was when Johnattan asked it to resolve a Git merge conflict in `features.py`. We checked that the default behaviour of the file did not change.
- **Responsibility.** The AI proposed and we decided what to keep. Every file in the repository was run on our machines before being committed.

---

## 2. Johnattan's part: evaluation, models, CLI and tests

### Concepts we asked the AI to explain

- the difference between the ROC curve and the PR curve, and why PR-AUC is our main metric;
- the role of `C` in logistic regression (regularisation), and the difference between log loss and hinge loss;
- how `class_weight="balanced"` and the decision threshold are two different ways of choosing an operating point.

### Planning

- A day-by-day plan up to the deadline, and a split of the work between the two of us.
- The order in which to write the functions of `evaluate.py` and `models.py`.

### File by file
| File | What the AI did | What Johnattan did |
|---|---|---|
| `evaluate.py` | Corrected `compute_metrics`, `get_oof_scores`, `cross_validate_model`, `find_best_threshold` and wrote the three plot functions; added support for `decision_function` (linear SVM) | Wrote the first versions, chose which metrics to keep, tested every function in the notebook |
| `models.py` | Guided function by function (names and signatures), then wrote a simplified version of the baseline/pipeline/comparison logic on request | Chose the models (logistic regression, KNN, linear SVM rather than models not seen in class), wrote and debugged `build_pipeline` |
| `config.py`, `config.yaml`, `pyproject.toml` | Drafted the files and the loading function | Adapted the keys to our code (`random_state`, `param_grids`) |
| `cli.py` | Wrote the command line interface (`train`, `evaluate`, `predict`) and its readable output | Adapted the imports to `data.py` and `models.py`, ran all three commands |
| `tests/` (evaluate, models, config, cli) | Drafted the tests and the shared `conftest.py` fixture | Fixed import errors, ran `pytest` until all tests passed |
| `models.ipynb` | Wrote some cells (comparison tables, PR curve overlay, threshold table, saving the model) and helped write the comments on the results | Ran the analysis and interpreted the results |

### Debugging and tooling
Virtual environment and `pip install -e` problems, import errors, a YAML `KeyError`, `pytest` fixtures not found, Git merge conflicts, and creating the `.gitignore`.

---

## 3. Yro's part: data, features, EDA, experiments and tests

### `data.py` (load, clean, split)
- **AI did**: proposed the three-function structure (`load_credit`, `clean`, `split_data`) 
- **Yro decided / changed**:
  - required `fetch_ucirepo` as in the sessions (the AI's first draft read a local CSV);
  - kept `ID` as the index;
  - kept every row: odd codes are merged (EDUCATION 0, 5, 6 → 4; MARRIAGE 0 → 3) and nothing is dropped;
  - chose a train/test split only (the AI first proposed 60/20/20);
  - added `prepare_data` and the error handling himself.
- **Checked by**: printing the shape (30000, 24) and the default rate (0.2212), as expected. Five tests in `test_data.py` check that cleaning drops nothing, that the split is stratified and reproducible, and that loading renames the columns.

### `features.py` (preprocessing and new features)
- **AI did**: based on the EDA findings, suggested one-hot encoding the `PAY_*` columns and five new features (`n_late`, `max_delay`, `util_1`, `util_avg`, `paid_share_1`).
- **Yro decided / changed**:
  - asked for `ColumnTransformer`, `StandardScaler` and `OneHotEncoder` as in the sessions, not a custom transformer class (the AI first suggested one);
  - wrote `add_features` in plain pandas;
  - kept the PAY codes -2, -1 and 0 untouched.
- **Rules accepted**: a bill ≤ 0 means nothing is owed, so `paid_share_1` = 1; the share is capped at 1.
- **Checked by**:
  - nine tests in `test_features.py`, each checking a new feature on a hand-made client whose answer is known;
  - an ablation study by cross-validation (see `experiments.ipynb`).
- **Known limitation**: the AI found a real weakness. A PAY value never seen in training is encoded like code -2 ("no consumption"). This is recorded as an expected-failure (`xfail`) test and has not been fixed yet.

### `test_data.py` and `test_features.py`
- **AI did**: wrote the tests, plus a stand-in runner to try them, because `pytest` could not be installed in its sandbox.
- **Yro decided**: to test only his two files, to ask for a shorter version, then to split it into two files instead of one.
- **Checked by**:
  - the AI broke the code on purpose in 7 ways, and each bug was caught by the right test;
  - in its sandbox: 13 passed + 1 expected failure;
  - on our machine, the full test suite (both parts) gives **33 passed, 1 expected failure**.

### `eda.ipynb`
- **AI did**: proposed the EDA cells (target balance, skewness, correlations, default rate per category and per bin).
- **Yro found / changed**: `describe()` printed means of categorical columns, which are meaningless, and the numerical features were never compared between defaulters and non-defaulters. The AI computed that comparison, and Yro changed the cell so it only summarises numerical columns. The EDA does not use the test set.

### `experiments.ipynb`
- **AI did**: wrote the experiment code (ablation with and without engineered features, `class_weight` vs SMOTE).
- **Yro decided**: which experiments to run, and to use only the three models from `models.py` (logistic regression with `max_iter=1000`, linear SVM, k-NN with k=50). SMOTE (`imbalanced-learn`) is not in the sessions; Yro asked for it explicitly.
- **Results**:
  - encoding PAY as categories is the main gain (ROC-AUC 0.727 → 0.769 for logistic regression);
  - `class_weight="balanced"` leaves ROC-AUC and PR-AUC unchanged and only moves the scores, like moving the threshold: it doubles the mean probability of the logistic regression (0.221 → 0.434);
  - k-NN benefits from the engineered features, while the linear models gain almost nothing from them once PAY is one-hot encoded;
  - SMOTE gives the same results as `class_weight` for the linear models (logistic regression: ROC-AUC 0.772, PR-AUC 0.539 in both cases) and lowers the PR-AUC of k-NN (0.522 → 0.510), so it brings no gain.

---

## 4. Where the AI was wrong, and what caught it

**Johnattan's part**
- It claimed KNN was about five times slower. Our timings did not support this.
- It said the tuned threshold would be well below 0.5. This is not true with `class_weight="balanced"`, which shifts the probabilities upwards.
- It made a wrong claim about how `find_best_threshold` behaves, and corrected it when questioned.

**Yro's part**
- **Loader and split**: the first draft ignored the sessions' `fetch_ucirepo`, used a 60/20/20 split and dropped `ID`. Yro corrected all three.
- **PAY codes**: the AI first presented -2 and 0 as documented statuses. When asked what they mean, it checked the data and corrected itself. Only -1 and 1–9 are documented; -2 ("no consumption") and 0 ("revolving credit") come from a community interpretation that the data supports. The codes were kept untouched.
- **Custom classes**: the AI suggested a custom transformer class, which is not in the sessions. Asking for the session tools instead led to the one-hot encoding of PAY, the biggest improvement of the project.
- **Import errors**: one AI-written file used an import that failed in our folder layout. Yro reported the error and it was fixed.
- **Changed recommendation**: the AI first advised against the engineered features, after testing them with logistic regression and a random forest. Re-tested with our three models, the conclusion changed for k-NN, which needs them. The features are therefore kept in `features.py` (`make_prep(engineered=True)`). The final logistic regression does not use them, because its gain is negligible.
- **Meaningless statistic**: the mean of categorical columns in the EDA was caught by Yro, not by the AI.

## 5. How we checked the AI's work

- We ran every step in our own notebooks and compared the output with what we expected (shapes, default rate, metrics).
- We questioned claims we did not understand, instead of accepting them.
- We used cross-validation on the training set only, and kept the test set untouched until the single final evaluation.
- We wrote unit tests for every module. Some bugs were introduced on purpose to confirm that the tests can fail.

## 6. What the AI did not decide

- The choice of the problem and the dataset.
- The scope and the split of the work between the two of us.
- The models: only models seen in class. We considered it more valuable to show that a known model does worse on this problem than to use a model we could not explain.
- The evaluation rules: PR-AUC as the main metric, a threshold that catches at least 60% of the defaulters without giving up precision entirely, and no resampling by default.
- The final model: logistic regression, chosen from our cross-validation results.
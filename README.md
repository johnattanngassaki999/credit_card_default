# Credit Card Default Prediction

Predicting which credit card clients will default on their next payment, from their profile and their last six months of bills and repayments.

**Authors**: NGASSAKI Johnattan · TIE Yro
**Course**: Machine Learning × Advanced Python Programming, Programme Grande École (2nd year), Aivancity

---

## 1. The problem

The dataset is the [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) dataset (id 350). It has 30,000 clients of a Taiwanese bank in 2005, 23 features, and a binary target: did the client default on the next month's payment?

This is an **imbalanced binary classification** problem: only **22.1%** of the clients default. So accuracy is not a useful metric, since a model that predicts "no default" for everyone is already 78% accurate. We evaluate the models with:

- **PR-AUC** (average precision), our main metric. It measures how well a model ranks the defaulters, and focuses on the minority class.
- **ROC-AUC**, a secondary ranking metric.
- **Precision, recall and F1** at a chosen decision threshold.

From a business point of view, missing a future defaulter (false negative) is the most expensive error. But flagging too many clients who would have repaid (false positives) makes the model useless. We therefore choose the threshold that **catches at least 60% of the defaulters** while keeping precision as high as possible.

## 2. Repository structure

```
credit_card_default/
├── src/credit_card_default/   # the Python package
│   ├── config.py              # loads config.yaml, defines PROJECT_ROOT
│   ├── data.py                # downloads, cleans and splits the data
│   ├── features.py            # preprocessing (ColumnTransformer) and engineered features
│   ├── models.py              # models, pipelines, tuning, comparison, save/load
│   ├── evaluate.py            # metrics, cross-validation, threshold choice, plots
│   └── cli.py                 # command line interface: train / evaluate / predict
├── tests/                     # unit tests (pytest)
├── notebooks/
│   ├── eda.ipynb              # exploratory data analysis
│   ├── experiments.ipynb      # preprocessing and class-imbalance experiments
│   └── models.ipynb           # model comparison, tuning, threshold, final evaluation
├── reports/                   # report.pdf, figures/, test_results.csv
├── models/                    # saved model (created by `train`)
├── config.yaml                # all the hard-coded values (seed, split, CV, grids, threshold rule)
├── clients_sample.csv         # 15 clients to try the `predict` command
├── pyproject.toml             # package metadata and dependencies
├── AI.md                      # how we used AI tools in this project
└── README.md
```

The code that the notebooks and the CLI share lives in the package. The notebooks only call its functions and show the results.

## 3. Installation

Requires **Python 3.10+** and an internet connection, because the data is downloaded from the UCI repository with `ucimlrepo` the first time it is used.

```bash
git clone <repository-url>
cd credit_card_default

python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

`-e` installs the package in editable mode, so code changes take effect without reinstalling. `[dev]` adds `pytest` and `ipykernel`, which you need to run the tests and the notebooks. To use the notebooks, select the `.venv` kernel in Jupyter or VS Code.

## 4. Usage

All the commands are run from the project root, with the virtual environment active. Every value they use (seed, test size, number of folds, hyperparameter grid, threshold rule) is read from `config.yaml`.

### Train the final model
```bash
python -m credit_card_default.cli train
```
This command:
1. tunes the regularisation parameter `C` of the logistic regression with a grid search (5-fold stratified CV, scored by PR-AUC) on the training set;
2. chooses the decision threshold on the out-of-fold scores, using the rule in `config.yaml`;
3. saves the pipeline and its threshold to `models/final_model.joblib`.

```
Training: logistic_regression
=============================
Training clients : 24,000
Best parameters  : C = 1
Threshold        : 0.478  (rule: min recall 0.6)

Out-of-fold metrics on the training set, at this threshold:
           value
precision  0.472
recall     0.602
f1         0.529
roc_auc    0.769
pr_auc     0.540

Model saved to   : models\final_model.joblib
```

### Evaluate on the test set
```bash
python -m credit_card_default.cli evaluate
```
This evaluates the saved model on the held-out test set (20% of the data, never used during model selection) and prints the metrics and the confusion matrix. **We run this command only once, on the final model**: if we used the test set to pick a model or a threshold, it would no longer give an unbiased estimate.

### Score new clients
```bash
python -m credit_card_default.cli predict clients_sample.csv --output predictions.csv
```
The input CSV must contain the same feature columns as the training data. The output contains the original columns plus two new ones: `score` (the predicted probability of default) and `flagged` (1 if the score is at least the threshold).

Once the package is installed, `credit-card-default train` (and the same for `evaluate` and `predict`) works too.

### Run the tests
```bash
python -m pytest
```

## 5. Method

1. **EDA** (`eda.ipynb`): class balance, distributions, undocumented category codes, how the default rate varies with each feature.
2. **Preprocessing** (`features.py`): numerical features are standardised and categorical features one-hot encoded (`handle_unknown="ignore"`), all inside a scikit-learn `Pipeline`. This way, the preprocessing is fitted on the training folds only, with no data leakage. The repayment-status columns `PAY_1`–`PAY_6` are treated as categories, not numbers (see section 6).
3. **Split**: stratified 80/20 train/test split with a fixed seed. Everything up to the final evaluation uses the training set only.
4. **Models**: a `DummyClassifier` (prior) as the baseline, then three models seen in class: **logistic regression**, **linear SVM** and **k-nearest neighbours**. We chose models we can explain rather than more powerful black boxes. Showing *why* a known model does better or worse on this problem was more valuable to us.
5. **Comparison**: 5-fold stratified cross-validation, PR-AUC and ROC-AUC (mean ± std), and the gain over the baseline.
6. **Tuning**: grid search over each model's main hyperparameter (`C` for logistic regression and linear SVM, `n_neighbors` for KNN).
7. **Class imbalance and threshold** (`experiments.ipynb`, `models.ipynb`): we compared no reweighting, `class_weight="balanced"` and SMOTE, then chose the operating point on out-of-fold scores.
8. **Final evaluation**: run once on the test set.

## 6. Results

### Exploratory analysis (key findings)
- 22.1% of the clients default.
- `EDUCATION` (345 rows) and `MARRIAGE` (54 rows) contain codes that the dataset documentation does not describe.
- The repayment status of the most recent month (`PAY_1`) is by far the strongest single predictor (correlation 0.33 with the target). The older months matter less and less (down to 0.19 for `PAY_6`).
- Defaulters have lower credit limits (median 90,000 vs 150,000 NT$) and repay smaller amounts.
- Bill and payment amounts are very right-skewed. Some bills are negative (about 500 per month), which means the client overpaid. 17–24% of the monthly payments are 0.

### Preprocessing and imbalance experiments (5-fold CV, training set)
- **Encoding `PAY_x` as categories is the most important preprocessing choice for the linear models**: logistic regression goes from ROC-AUC 0.727 / PR-AUC 0.507 with numeric `PAY_x` to 0.769 / 0.543 with one-hot encoding. The codes (-2, -1, 0, 1…8) are not on a linear scale, and one-hot encoding lets each status have its own effect.
- **Engineered features** (number of late months, worst delay, credit utilisation, share of the last bill repaid) help a lot when `PAY_x` is numeric. Once `PAY_x` is one-hot encoded, they add almost nothing (ROC-AUC +0.003, PR-AUC unchanged). We therefore keep the **base features** for the final model.
- **`class_weight="balanced"` and SMOTE do not improve the ranking** (same ROC-AUC and PR-AUC). They only shift the scores upwards: at the default threshold of 0.5, recall goes from 0.36 to about 0.61. Moving the threshold has the same effect. SMOTE also adds complexity and gives no extra gain.

### Model comparison (5-fold CV, training set)

| Model | PR-AUC (mean ± std) | ROC-AUC (mean ± std) | PR-AUC gain vs dummy |
|---|---|---|---|
| **Logistic regression** | **0.542 ± 0.012** | **0.770 ± 0.006** | **+0.321** |
| Linear SVM | 0.539 ± 0.013 | 0.767 ± 0.006 | +0.318 |
| KNN (k = 50) | 0.509 ± 0.017 | 0.752 ± 0.012 | +0.288 |
| Dummy (baseline) | 0.221 ± 0.000 | 0.500 ± 0.000 | — |

- Logistic regression and linear SVM are tied: the gap is much smaller than one standard deviation, and both are linear models.
- KNN is clearly behind, and less stable across folds.
- We keep **logistic regression**: it outputs probabilities, its coefficients are interpretable, and it is the simplest of the two best models.

### Final model
- Logistic regression with `class_weight="balanced"`, `C = 1`, and the base features with `PAY_x` one-hot encoded.
- Threshold **0.478**: the threshold with the best precision among those with a recall of at least 0.60, chosen on out-of-fold scores.
- Out-of-fold metrics at this threshold: precision 0.472, recall 0.602, F1 0.529. We catch 6 defaulters out of 10, and about 1 flagged client in 2 really defaults.

### Test set (evaluated once)

| Metric | Value |
|---|---|
| Precision | _[to fill]_ |
| Recall | _[to fill]_ |
| F1 | _[to fill]_ |
| ROC-AUC | _[to fill]_ |
| PR-AUC | _[to fill]_ |

_[One sentence: are the test results close to the cross-validation results?]_

## 7. Limitations

- The data comes from a single bank in Taiwan in 2005. The model would need to be retrained and checked before being used on other clients.
- With a PR-AUC around 0.54, the features only partly explain default. About half of the flagged clients still repay, so the model should support human decisions, not replace them.
- Some category codes are undocumented, so we kept them as observed instead of guessing what they mean.
- `SEX`, `AGE` and `MARRIAGE` are used as features. A real credit-scoring system would have to check fairness and legal constraints before using them.

## 8. Team

| | Johnattan NGASSAKI | Yro TIE |
|---|---|---|
| Modules | `evaluate.py`, `models.py`, `cli.py` | `data.py`, `features.py` |
| Notebooks | `models.ipynb` | `eda.ipynb`, `experiments.ipynb` |
| Tests | evaluate, models, config, cli | data, features |

## 9. Use of AI

See [AI_usage.md](AI_usage.md).
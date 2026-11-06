# Sources and experimental scope

## Course references

- **Description - Project Heart Disease Diagnosis:** 49 pages; data schema,
  four feature variants, seven classifiers, evaluation, TreeSHAP and local demo.
- **Slide-v1 - Project Heart Disease Diagnosis:** 181 pages; visual overview and
  supplementary KMeans/stacking models.

The course PDFs are external learning references and are not bundled in this
repository. Commands inside those documents are examples, not setup instructions
for this repository. Follow README.md for the maintained local workflow.

## Data

The single canonical input is `data/raw/cleveland.csv`, a headerless Cleveland
file with 303 rows, 13 input columns and the original 0-4 target. Missing values
are represented by `?`. The loader performs binary target conversion at runtime.
See [NOTICE.md](../NOTICE.md) for source, attribution and dataset license.

## Mapping from Description to implementation

| Reference section | Implementation |
|---|---|
| Pages 4-8: goals, schema, EDA | config.py, data.py, eda.py, notebook 00 |
| Pages 9-10: stratified 80:10:10, Original | data.split_dataset, preprocessing.py |
| Pages 11-13: FE, one-hot, MI | features.py, selection.py, notebook 01 |
| Pages 14-16: DT Top-10, CSV/ZIP | selection.py, artifacts.export_variants |
| Pages 17-32: seven classifiers and CV | models.py, pipelines.py, training.py |
| Pages 33-38: TreeSHAP | explain.py, notebook 03 |
| Pages 39-40: evaluation and stability | evaluation.py, reports/final_report.md |
| Pages 41-45: demo | app.py, inference.py |

## Methodological choices

- FE uses MI Top-13; DT selection uses Top-10. These are separate settings.
- Every learned preprocessing/selection step is inside the CV pipeline. Exported
  CSVs are for inspection, not the inputs to a label-informed CV search.
- Core selection is locked before test reporting. KMeans/stacking were added
  afterward and are labeled exploratory_post_hoc; they do not replace the demo model.
- Repeated 5-fold x 3 CV on train describes stability of already chosen settings;
  it is not nested CV of the full model-selection procedure.
- RF SHAP uses tree-path-dependent explanations. The RF raw class output is
  checked against predict_proba; explanation values refer to transformed features.

## Primary API documentation

- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
- [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
- [Ruff configuration](https://docs.astral.sh/ruff/configuration/)
- [GitHub checkout action](https://github.com/actions/checkout)
- [GitHub setup-python action](https://github.com/actions/setup-python)

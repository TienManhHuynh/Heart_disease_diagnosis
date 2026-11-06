"""TreeSHAP RF/DT: tree-path-dependent và kiểm chứng xác suất lớp 1."""

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_json
from .evaluation import disease_probability


def explain_tree_pipeline(fitted_pipeline, X_raw, output_dir, y_true=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = fitted_pipeline.named_steps["features"]
    model = fitted_pipeline.named_steps["model"]
    X = features.transform(X_raw)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    if not isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
        raise ValueError("Luồng SHAP hiện hỗ trợ RandomForestClassifier/DecisionTreeClassifier.")
    # Với RF/DT classifier, raw tree output là xác suất từng lớp. Dùng số mẫu
    # trên nhánh của cây đã học làm phân phối tham chiếu, không lấy test làm background.
    explainer = shap.TreeExplainer(
        model, model_output="raw", feature_perturbation="tree_path_dependent"
    )
    explanation = explainer(X)
    class_index = list(model.classes_).index(1)
    if explanation.values.ndim == 3:
        disease = explanation[..., class_index]
    elif explanation.values.ndim == 2 and class_index == 1:
        disease = explanation
    else:
        raise ValueError(f"Unsupported SHAP output shape: {explanation.values.shape}")
    probability = disease_probability(fitted_pipeline, X_raw)
    reconstructed = disease.base_values + disease.values.sum(axis=1)
    max_error = float(np.max(np.abs(reconstructed - probability)))
    if max_error > 1e-5:
        raise ValueError(f"SHAP additivity mismatch: {max_error}")
    prediction = fitted_pipeline.predict(X_raw)
    candidates = (
        np.flatnonzero((np.asarray(y_true) == 1) & (prediction == 1))
        if y_true is not None
        else np.array([], dtype=int)
    )
    if len(candidates):
        local_id = int(candidates[np.argmax(probability[candidates])])
        rule = "Highest disease probability among correctly predicted positives; illustrative, not representative."
    else:
        local_id = int(np.argmax(probability))
        rule = "No correctly predicted positives; highest probability sample used."
    plots = [
        ("bar.png", lambda: shap.plots.bar(disease, show=False)),
        ("beeswarm.png", lambda: shap.plots.beeswarm(disease, show=False)),
        ("waterfall.png", lambda: shap.plots.waterfall(disease[local_id], show=False)),
    ]
    for name, draw in plots:
        draw()
        fig = plt.gcf()
        fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
    importance = pd.Series(
        np.abs(disease.values).mean(axis=0), index=X.columns, name="mean_abs_shap"
    ).sort_values(ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv")
    result = {
        "class": 1,
        "output_scale": "probability",
        "explainer_model_output": "raw",
        "feature_perturbation": "tree_path_dependent",
        "reference": "Training path counts embedded in the fitted forest/tree",
        "max_additivity_error": max_error,
        "sample_count": len(X),
        "feature_count": X.shape[1],
        "local_position": local_id,
        "local_original_index": int(X_raw.index[local_id]),
        "selection_rule": rule,
        "probability": float(probability[local_id]),
        "reconstructed_probability": float(reconstructed[local_id]),
        "features_are_transformed": True,
    }
    write_json(output_dir / "explanation.json", result)
    return result

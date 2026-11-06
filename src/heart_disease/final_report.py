"""Báo cáo, phân tích sai số và nhận xét notebook từ artifact đã khóa."""

import json
import math

import numpy as np
import pandas as pd

from .artifacts import write_json
from .config import PROJECT_ROOT
from .evaluation import disease_probability
from .models import parameter_grid
from .workflows import _load_verified, _pipeline


def markdown_table(frame):
    frame = frame.copy()
    rows = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append(
            "| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in values) + " |"
        )
    return "\n".join(rows)


def score_table(frame):
    out = frame[["model", "variant"]].copy()
    out["CV acc ± std"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(frame.cv_mean, frame.cv_std)]
    for name, label in (
        ("val_accuracy", "Val acc"),
        ("test_accuracy", "Test acc"),
        ("test_f1", "Test F1"),
        ("test_recall", "Recall 1"),
    ):
        out[label] = frame[name].map(lambda x: f"{x:.3f}")
    return markdown_table(out)


def build_final_report(settings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shap

    manifest, splits = _load_verified(settings)
    root = PROJECT_ROOT
    reports = root / "reports"
    table = pd.read_csv(reports / "results.csv")
    if set(table.run_id) != {manifest["run_id"]}:
        raise ValueError("results.csv không thuộc run hiện tại.")
    output = reports / "runs" / manifest["run_id"]
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    details = json.loads((output / "classification_reports.json").read_text(encoding="utf-8"))
    features = json.loads(
        (root / "data/processed/feature_manifest.json").read_text(encoding="utf-8")
    )
    extension_dir = reports / "extensions" / manifest["run_id"]
    if not (extension_dir / "results.csv").exists():
        raise FileNotFoundError("Chạy extensions trước khi tạo báo cáo hoàn chỉnh.")
    extension = pd.read_csv(extension_dir / "results.csv")
    selected_model, selected_variant = manifest["selected_key"].split("/")
    chosen = table[(table.model == selected_model) & (table.variant == selected_variant)].iloc[0]
    pipeline = _pipeline(manifest, manifest["selected_key"])
    X, y = splits["test"]
    prediction = pipeline.predict(X)
    probability = disease_probability(pipeline, X)
    predictions = X.copy()
    predictions.insert(0, "source_row_1based", X.index + 1)
    predictions["true_label"], predictions["prediction"] = y, prediction
    predictions["disease_probability"] = probability
    predictions["error_type"] = [
        "TP" if a == b == 1 else "TN" if a == b == 0 else "FN" if a == 1 else "FP"
        for a, b in zip(y, prediction)
    ]
    predictions.to_csv(reports / "test_predictions.csv", index_label="original_index")
    errors = predictions[predictions.error_type.isin(["FP", "FN"])]
    errors.to_csv(reports / "error_analysis.csv", index_label="original_index")
    examples = predictions.loc[[62, 101, 66]].copy()
    examples.to_csv(reports / "demo_examples.csv", index_label="original_index")

    # SHAP analysis is descriptive after the locked test report; no tuning.
    rf = _pipeline(manifest, "random_forest/original")
    transformed = rf.named_steps["features"].transform(X)
    explanation = shap.TreeExplainer(
        rf.named_steps["model"], model_output="raw", feature_perturbation="tree_path_dependent"
    )(transformed)[..., 1]
    shap_info = json.loads((root / "artifacts/metadata/shap.json").read_text(encoding="utf-8"))
    if shap_info["run_id"] != manifest["run_id"]:
        raise ValueError("SHAP không thuộc run hiện tại.")
    local_id = shap_info["local_position"]
    local = pd.DataFrame(
        {
            "feature": transformed.columns,
            "transformed_value": transformed.iloc[local_id].to_numpy(),
            "shap_value": explanation.values[local_id],
        }
    )
    local = local.iloc[np.argsort(-np.abs(local.shap_value.to_numpy()), kind="stable")]
    local.to_csv(reports / "shap_local_contributions.csv", index=False)
    importance = pd.Series(np.abs(explanation.values).mean(axis=0), index=transformed.columns)
    importance = importance.sort_values(ascending=False)
    directions = []
    for j, column in enumerate(transformed.columns):
        rho = pd.Series(transformed[column].to_numpy()).corr(
            pd.Series(explanation.values[:, j]), method="spearman"
        )
        directions.append(
            {"feature": column, "spearman_value_vs_shap": None if pd.isna(rho) else float(rho)}
        )
    analysis = {
        "run_id": manifest["run_id"],
        "selected_key": manifest["selected_key"],
        "confusion_matrix": details[manifest["selected_key"]]["metrics"]["confusion_matrix"],
        "error_counts": predictions.error_type.value_counts().to_dict(),
        "shap_base_value": float(explanation.base_values[local_id]),
        "shap_local_probability": shap_info["probability"],
        "shap_local_index": int(X.index[local_id]),
        "shap_directions_descriptive": directions,
        "core_rows": len(table),
        "extension_rows": len(extension),
    }
    write_json(reports / "analysis.json", analysis)

    # Figures make model and variant comparisons easier to inspect.
    figures = reports / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, value, title in zip(
        axes, ("cv_mean", "test_accuracy"), ("Train CV accuracy", "Test accuracy (descriptive)")
    ):
        pivot = table.pivot(index="model", columns="variant", values=value)
        sns.heatmap(
            pivot, annot=True, fmt=".3f", vmin=0.70, vmax=1.0, cmap="Blues", ax=ax, cbar=False
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    stability = evaluation["stability_training_only"]
    ax.plot(range(1, len(stability["scores"]) + 1), stability["scores"], "o-", color="#167d9a")
    ax.axhline(
        stability["mean"],
        color="#ce6852",
        linestyle="--",
        label=f"Mean F1 = {stability['mean']:.3f}",
    )
    ax.set(
        xlabel="Repeated CV fold",
        ylabel="Disease-class F1",
        ylim=(0.6, 1),
        title="Stability on TRAIN only; fixed selected configuration",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "stability.png", dpi=150)
    plt.close(fig)

    train_X, train_y = splits["train"]
    raw = pd.concat(
        [values.assign(target=labels) for values, labels in splits.values()]
    ).sort_index()
    numeric_stats = train_X[["age", "trestbps", "chol", "thalach", "oldpeak"]].describe().T
    numeric_stats = (
        numeric_stats[["mean", "std", "min", "50%", "max"]].round(2).reset_index(names="Feature")
    )
    numeric_stats.columns = ["Feature", "Mean", "Std", "Min", "Median", "Max"]
    split_rows = []
    for name, (values, labels) in splits.items():
        split_rows.append(
            {
                "Tập": name,
                "Số mẫu": len(values),
                "Lớp 0": int((labels == 0).sum()),
                "Lớp 1": int((labels == 1).sum()),
            }
        )
    grids = pd.DataFrame(
        [
            {"Model": name, "Tham số quét": str(parameter_grid(name)).replace("model__", "")}
            for name in settings.models
        ]
    )
    feature_rows = pd.DataFrame(
        [
            {
                "Biến thể": name,
                "Số cột": len(record["columns"]),
                "Tên feature": ", ".join(record["columns"]),
            }
            for name, record in features.items()
        ]
    )
    rf_rows = table[table.model == "random_forest"]
    orig = table[table.variant == "original"].set_index("model")
    fe = table[table.variant == "fe"].set_index("model")
    fe_improved = int((fe.cv_mean > orig.cv_mean).sum())
    dt = table[table.variant == "original_dt"].set_index("model")
    dt_improved = int((dt.cv_mean > orig.cv_mean).sum())
    confusion = np.asarray(analysis["confusion_matrix"])
    tn, fp, fn, tp = confusion.ravel()
    n, correct = len(y), int((y.to_numpy() == prediction).sum())
    z = 1.959963984540054
    center = (correct / n + z * z / (2 * n)) / (1 + z * z / n)
    radius = (
        z * math.sqrt((correct / n) * (1 - correct / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    )
    local_table = local.head(6).copy()
    local_table["transformed_value"] = local_table.transformed_value.map(
        lambda value: f"{value:.4f}"
    )
    local_table["shap_value"] = local_table.shap_value.map(lambda value: f"{value:+.4f}")
    error_table = errors[
        ["source_row_1based", "true_label", "prediction", "disease_probability", "error_type"]
    ].copy()
    error_table.disease_probability = error_table.disease_probability.map(
        lambda value: f"{value:.4f}"
    )
    example_table = examples[
        ["source_row_1based", "true_label", "prediction", "disease_probability", "error_type"]
    ].copy()
    example_table.disease_probability = example_table.disease_probability.map(
        lambda value: f"{value:.4f}"
    )
    ranking = pd.DataFrame(
        {"Feature": importance.head(6).index, "Mean abs SHAP": importance.head(6).round(4).values}
    )
    version_table = pd.DataFrame(
        list(manifest["versions"].items()), columns=["Thư viện", "Phiên bản"]
    )
    report = f"""# Heart Disease Diagnosis

## Báo cáo hoàn chỉnh - Project AI 3.1

Bài toán: phân loại nhị phân bệnh tim trên Cleveland. Báo cáo sử dụng số đo thực từ run `{manifest["run_id"]}` và phần mở rộng được ghi riêng. Không lấy kết quả minh họa trong tài liệu làm số đo của project.

**Kết quả chính.** Quy tắc chọn theo CV/validation chọn `{manifest["selected_key"]}` với CV accuracy {chosen.cv_mean:.4f} ± {chosen.cv_std:.4f}. Trên test: accuracy {chosen.test_accuracy:.4f}, F1 lớp bệnh {chosen.test_f1:.4f}, recall lớp bệnh {chosen.test_recall:.4f}. Test có {n} mẫu; đây là kết quả học tập trên dữ liệu nhỏ.

**Phạm vi hoàn thành.** Đã triển khai bốn biến thể; bảy classifier chính (28 cấu hình); KMeans và stacking (8 cấu hình bổ sung); EDA; feature selection; test/error analysis; TreeSHAP; demo local; kiểm thử và hướng dẫn tái lập.

**Cách đọc.** Phần 1-3 mô tả dữ liệu/phương pháp, phần 4-6 trình bày kết quả và diễn giải, phần 7 ghi thí nghiệm mở rộng, phần 8-10 bàn giao và giới hạn. Bảng số liệu chi tiết ở results.csv, test_predictions.csv và thư mục artifact.

<!-- pagebreak -->

## 1. Dữ liệu, chất lượng và EDA

Dữ liệu local có {len(raw)} bệnh nhân, 13 feature và target. Nhãn gốc 0 giữ lớp 0; các nhãn 1-4 gộp thành lớp 1. Sau chuyển đổi có 164 mẫu lớp 0 (54,13%) và 139 mẫu lớp 1 (45,87%), mức chênh lệch chưa lớn. File có 4 giá trị thiếu ở ca và 2 ở thal; không có dòng trùng hoàn toàn. Giữ cả 303 mẫu, không xóa dòng thiếu; imputer được fit trên training subset.

{markdown_table(pd.DataFrame(split_rows))}

Thống kê bên dưới chỉ dùng train. Tuổi trung vị 56; huyết áp nghỉ trung vị 130; cholesterol trung vị 244,5. Cholesterol tối đa 564 và oldpeak tối đa 6,2 tạo đuôi phải; chưa có bằng chứng xác nhận lỗi nhập nên không tự loại hoặc cắt ngưỡng. Các cột cp/thal/slope là mã nhóm, không nên diễn giải như thang đo liên tục có khoảng cách y khoa đồng đều.

{markdown_table(numeric_stats)}

![Phân bố năm biến numeric trên train](figures/train_numeric.png)

<!-- pagebreak -->

## 2. Bốn biến thể và tránh rò rỉ dữ liệu

Original dùng median + StandardScaler cho năm numeric; most_frequent + MinMaxScaler cho tám categorical. FE bổ sung chol/age, trestbps/age, thalach/age và age_bin từ quantile tuổi train; one-hot categorical; bỏ cột hằng theo train và giữ MI Top-13. Các biến thể có DT chọn Top-10 bằng feature_importances_.

{markdown_table(feature_rows)}

MI đo phụ thuộc từng feature với nhãn; DT importance phụ thuộc các split của một cây và có thể chia sẻ thông tin giữa feature tương quan. Ví dụ exang_0.0 và exang_1.0 có cùng MI vì là hai mã bổ sung của cùng một biến; việc giữ cả hai không tạo hai bằng chứng độc lập. Trong Original+DT, trestbps/fbs/restecg không thuộc Top-10, không có nghĩa ba biến vô ích cho mọi mô hình.

Tất cả pipeline được fit lại trong mỗi training fold: imputer, scaler, ngưỡng age_bin, one-hot, MI và DT selector. Sau fit chỉ gọi transform trên val/test, giữ tên cột và thứ tự. 12 CSV được xuất để kiểm tra/nộp; CV vẫn nhận dữ liệu thô để tránh chọn feature theo nhãn toàn train trước CV. Feature trong bảng là kết quả fit cuối trên toàn train; các fold có thể chọn danh sách khác.

FE đồng thời thay mã hóa, tạo feature và chọn feature. Do đó nếu điểm thay đổi, chưa thể quy nguyên nhân riêng cho các tỉ lệ theo tuổi; cần ablation riêng để kết luận từng thành phần.

## 3. Giao thức benchmark

Chia stratified 80:10:10, seed {settings.seed}; giữ index bệnh nhân gốc, các tập không giao nhau. Mỗi cấu hình dùng Stratified {settings.cv_folds}-fold CV trên train, scoring accuracy, refit best trên toàn train. Không refit trên val/test. Thứ tự chọn đã đặt trước: CV mean cao nhất, hòa điểm dùng validation F1, sau đó tên model/variant. Test chỉ dùng khi báo cáo cho danh sách cấu hình đã khóa.

{markdown_table(grids)}

Các thiết lập cố định đáng chú ý: RF max_depth=5; GradientBoosting max_depth=2; XGBoost max_depth=3, learning_rate=0.05; estimator có hỗ trợ random_state dùng seed 42. Lưới được giữ nhỏ để chạy tái lập trên máy cá nhân, không khẳng định tối ưu toàn cục. Tham số tốt nhất từng cấu hình và điểm từng ứng viên lưu trong metadata/cv_results.csv.

<!-- pagebreak -->

## 4. Kết quả chính và so sánh

{score_table(table)}

Trong bảy mô hình, FE tăng CV mean so với Original ở {fe_improved}/7 trường hợp; Original+DT tăng ở {dt_improved}/7. KNN và GradientBoosting hưởng lợi về CV khi chuyển Original sang FE, trong khi RF không tăng. Với RF, Original có CV {chosen.cv_mean:.4f}; FE có CV {float(rf_rows[rf_rows.variant == "fe"].cv_mean.iloc[0]):.4f}. Test giảm từ 0,9032 xuống 0,8710 khi chuyển RF Original sang FE hoặc Original+DT, và xuống 0,8387 ở FE+DT.

Kết quả cho thấy tạo/chọn feature không cải thiện nhất quán. Với tập nhỏ, one-hot và MI có thể thêm dư thừa hoặc loại feature có ích khi kết hợp; RF vốn đã chọn feature khi chia cây. Đây là các diễn giải phù hợp quan sát, chưa phải bằng chứng xác định cơ chế gây thay đổi. Baseline NB Original có test 0,8387, vì vậy lợi ích của RF cần đặt trong bối cảnh mô hình đơn giản đã có tín hiệu dự đoán.

## 4.1. So sánh trực quan và độ dao động

Heatmap dùng cùng thang màu để đối chiếu model/variant. Cột test chỉ mô tả các cấu hình đã khóa, không dùng để chọn lại mô hình. Biểu đồ repeated CV phía dưới chỉ dùng train và metric F1 lớp 1; các fold không độc lập hoàn toàn vì mẫu được tái sử dụng giữa các lần lặp.

![So sánh CV và test; màu dùng cùng thang đo](figures/model_comparison.png)

![F1 qua 15 fold repeated CV của cấu hình chính](figures/stability.png)

Dao động F1 từ khoảng 0,70 đến 0,90 cho thấy kết quả phụ thuộc vào tập mẫu được giữ lại mỗi fold. Không chọn fold hoặc lần chia có điểm cao nhất làm kết quả đại diện. Chênh lệch nhỏ giữa các cấu hình cần được đọc cùng với biến động này và số mẫu ít.

<!-- pagebreak -->

## 5. Sai số, recall và độ ổn định

Confusion matrix của cấu hình đã chọn (hàng là nhãn thật, cột là dự đoán):

{markdown_table(pd.DataFrame({"Nhãn thật": ["0 - không bệnh", "1 - có bệnh"], "Dự đoán 0": [int(tn), int(fn)], "Dự đoán 1": [int(fp), int(tp)]}))}

Có {int(tp)} true positive, {int(tn)} true negative, {int(fp)} false positive và {int(fn)} false negative. Recall lớp 1 = {int(tp)}/{int(tp + fn)} = {chosen.test_recall:.4f}; precision lớp 1 = {int(tp)}/{int(tp + fp)} = {chosen.test_precision:.4f}. Accuracy cao không loại bỏ việc bỏ sót: dòng CSV 67 có nhãn 1 nhưng dự đoán 0, xác suất lớp 1 khoảng 0,3862.

{markdown_table(error_table)}

Mẫu false negative có age=60, cp=3, thalach=155, oldpeak=3, ca=0, thal=3. Các đặc trưng có thể đưa tín hiệu khác chiều; quan sát một mẫu không đủ xác nhận nguyên nhân y khoa hoặc lỗi nhãn. Hai false positive ở dòng 93 và 272 cho thấy xác suất mô hình có thể cao nhưng vẫn sai. Giữ nguyên quy tắc quyết định đã fit; không chọn ngưỡng lại dựa trên ba lỗi test này.

![Confusion matrix của mô hình đã chọn](runs/{manifest["run_id"]}/confusion_{selected_model}_{selected_variant}.png)

Test chỉ có {n} mẫu; một lỗi làm accuracy đổi {100 / n:.2f} điểm phần trăm. Khoảng Wilson 95% cho accuracy {correct}/{n} xấp xỉ [{center - radius:.3f}; {center + radius:.3f}], cho thấy bất định đáng kể. Khoảng này giả định các mẫu độc lập, không đo đầy đủ sai lệch do chọn mô hình hoặc khác biệt quần thể.

Repeated 5-fold x 3 trên TRAIN, với hyperparameter đã chọn, cho F1 trung bình {stability["mean"]:.4f}, std {stability["std"]:.4f}. Đây là độ dao động mô tả trên cùng train, không phải nested CV hay một ước lượng hoàn toàn độc lập của toàn bộ quá trình lựa chọn. Không so trực tiếp F1 này với CV accuracy như cùng một metric.

<!-- pagebreak -->

## 6. TreeSHAP: toàn cục và một dự đoán

Giải thích Random Forest + Original, lớp 1, dùng tree-path-dependent: phân phối tham chiếu là số mẫu trên nhánh cây đã fit. Với RF classifier, raw tree output tương ứng xác suất từng lớp; code kiểm chứng trực tiếp với predict_proba. SHAP không phải classifier bổ sung và không chứng minh quan hệ nhân quả.

{markdown_table(ranking)}

Trên 31 mẫu giải thích, cp, thal và ca có mean absolute SHAP cao nhất; tiếp theo oldpeak và thalach. Mean absolute chỉ cho biết độ lớn, không có dấu. Beeswarm cho thấy một số giá trị mã cp/thal/ca cao thường nằm phía đóng góp dương trong nhóm mẫu này; vì đây là mã categorical và sample nhỏ, không suy diễn “mỗi đơn vị tăng” thành tác động liên tục hoặc kết luận cho mọi bệnh nhân.

![Beeswarm; màu thể hiện giá trị sau biến đổi](runs/{manifest["run_id"]}/shap/beeswarm.png)

Mẫu local có index gốc {analysis["shap_local_index"]} (dòng CSV {analysis["shap_local_index"] + 1}), age=58, cp=4, thal=7, ca=3, oldpeak=2,2, thalach=131. Đây là true positive có xác suất cao nhất trong số true positive, được chọn làm ví dụ dễ quan sát; không đại diện ngẫu nhiên cho toàn tập.

{markdown_table(local_table)}

Giá trị nền {analysis["shap_base_value"]:.4f} cộng tổng đóng góp thành {shap_info["probability"]:.4f}, đúng xác suất lớp 1; sai số lớn nhất toàn tập {shap_info["max_additivity_error"]:.2e}. Trong mẫu này thal, cp, ca, oldpeak làm tăng đầu ra; thalach sau chuẩn hóa thấp hơn mức trung bình train cũng có đóng góp dương. Các trị số hiển thị trong waterfall là sau preprocessing, không còn đơn vị huyết áp/nhịp tim gốc.

![Waterfall cho mẫu minh họa](runs/{manifest["run_id"]}/shap/waterfall.png)

![Mức ảnh hưởng toàn cục theo mean absolute SHAP](runs/{manifest["run_id"]}/shap/bar.png)

<!-- pagebreak -->

## 7. Phần mở rộng: KMeans và stacking

Các thí nghiệm này được bổ sung sau khi đã xem test của benchmark chính, nên là phân tích khám phá hậu nghiệm, không phải so sánh xác nhận độc lập. Chúng không thay đổi selected_pipeline hoặc quy tắc lựa chọn chính. Cùng split/index, pipeline fit trong training fold, grid khóa trước khi chạy test mở rộng.

KMeans dùng hai cluster, n_init quét 10/20. Nhãn cluster được ánh xạ bằng tần suất lớp trong chính training subset; xác suất được làm trơn bằng prior train (smoothing=1). Đây là điểm tần suất theo cluster, không phải xác suất bệnh đã hiệu chỉnh. Không dùng nhãn val/test để ánh xạ cluster.

Stacking kết hợp KNN k=5, DT max_depth=5 và GaussianNB. Mỗi base learner có feature pipeline riêng; OOF 3-fold tạo đầu vào cho LogisticRegression. Outer CV 5-fold chọn C của meta learner trong 0,1/1,0. Không tiền xử lý toàn training subset trước OOF và không dùng dự đoán in-sample để học meta model.

{score_table(extension)}

KMeans có test cao hơn mức CV trong một số biến thể, nhưng CV thấp và test nhỏ nên chưa có cơ sở xem clustering tốt hơn classifier chính. Stacking Original+DT có CV khoảng 0,8224, gần RF Original; test các stacking ở mức 0,8387. Thêm tầng mô hình không tự động tạo cải thiện. Do các thí nghiệm được bổ sung hậu nghiệm, các so sánh này chỉ dùng để hiểu phương pháp.

<!-- pagebreak -->

## 8. Demo và tái lập

Demo Streamlit nạp nguyên pipeline đã lưu, kiểm tra SHA-256 metadata, nhận 13 cột raw theo schema, gọi predict/predict_proba; không train lại khi submit. Xác suất là đầu ra của mô hình, chưa được hiệu chỉnh hay kiểm định lâm sàng.

{markdown_table(example_table)}

Các ví dụ là hàng của test đã báo cáo, dùng để minh họa hành vi, không phải mẫu kiểm định mới. Toàn bộ 13 giá trị đầu vào có trong demo_examples.csv. Bảng trên là bằng chứng kiểm thử đầu vào/đầu ra của demo; các trị số xác suất được tính trực tiếp từ pipeline đã lưu.

Chạy local: `.venv/Scripts/python.exe -m streamlit run app.py`. Kiểm tra code: `.venv/Scripts/python.exe scripts/check_project.py`. Chạy lại từ đầu khi chủ động tạo thực nghiệm mới: `python -m heart_disease run`, sau đó `extensions` và `report`. Báo cáo PDF có script export riêng; xem README. Các notebook đã có nhận xét dựa trên run này, cần tạo lại report khi đổi dữ liệu/cấu hình.

## 9. Hạn chế, kết luận và hướng phát triển

Trong giao thức đã đặt, RF Original là cấu hình được chọn, đạt accuracy 0,9032 và recall lớp 1 0,9286 trên 31 mẫu test. Tuy nhiên dữ liệu nhỏ và lịch sử, một nguồn Cleveland, chưa có external validation; mọi kết luận chỉ áp dụng trong phạm vi thí nghiệm này. Lưới tham số nhỏ; chưa có nested CV của toàn bộ quá trình chọn model, ablation từng bước FE, phân tích theo nhóm hoặc đánh giá calibration. Đây là hướng phát triển, không phải những bước đã thực hiện.

Nếu tiếp tục nghiên cứu, đặt trước metric/không gian tìm kiếm, dùng nested/repeated CV hoặc tập đánh giá mới; không đổi seed/chọn ngưỡng theo test đã xem. Với bài học hiện tại, giá trị chính là quy trình tái lập, tránh rò rỉ dữ liệu, đánh giá nhiều metric và giải thích đúng giới hạn của mô hình.

<!-- pagebreak -->

## 10. Nguồn và phiên bản

1. Tài liệu môn học: Description - Project Heart Disease Diagnosis, các phần dữ liệu, bốn biến thể, benchmark, TreeSHAP và demo.
2. Slide-v1 - Project Heart Disease Diagnosis, pipeline và các mô hình mở rộng. Đường dẫn local được ghi trong docs/SOURCES.md.
3. UCI Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease (nguồn bộ Cleveland; project sử dụng CSV local đã có).
4. scikit-learn common pitfalls: https://scikit-learn.org/stable/common_pitfalls.html
5. StackingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
6. SHAP TreeExplainer: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
7. Streamlit AppTest: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest

{markdown_table(version_table)}

SHA-256 CSV gốc: `{manifest["raw_sha256"]}`. Phiên bản đầy đủ nằm trong requirements-lock.txt. Metadata lưu seed, split index, config hash và model hash. Báo cáo này là bản phân tích được tạo từ artifact thực; người nộp chịu trách nhiệm đọc, hiểu và đối chiếu yêu cầu hình thức của môn học.
"""
    (reports / "final_report.md").write_text(report, encoding="utf-8")
    narratives = {
        "00_eda.ipynb": f"## Nhận xét EDA đã hoàn thiện\n\nRun `{manifest['run_id']}` có 303 mẫu, 164 lớp 0 và 139 lớp 1. Có 4 ô thiếu ở ca, 2 ở thal và 0 dòng trùng hoàn toàn. Giữ 303 dòng và impute trong training fold. Tuổi train trung vị 56; chol trung vị 244,5 nhưng cực đại 564, oldpeak có đuôi phải. Không tự loại giá trị lớn khi chưa xác nhận lỗi. Tám categorical là mã nhóm; giữ cách xử lý đúng từng biến thể. Chia tập 242/30/31 có phân tầng và index không trùng. Xem final_report.md phần 1 để đọc bảng và lập luận đầy đủ.",
        "01_feature_variants.ipynb": "## Nhận xét về đặc trưng đã hoàn thiện\n\nOriginal sau ColumnTransformer có năm numeric trước, tám categorical sau, không giữ nguyên thứ tự CSV. Original và FE có 13 cột; hai biến thể DT có 10 cột. Original+DT bỏ trestbps/fbs/restecg khỏi Top-10 trên train cuối. FE giữ thal_3.0, thal_7.0, cp_4.0, ca_0.0 và các feature khác theo MI. exang_0/exang_1 có cùng MI vì hai mã bổ sung, không là hai nguồn thông tin độc lập. MI và DT đo khác nhau nên thứ hạng có thể lệch. Danh sách có thể thay đổi giữa fold; không chọn lại theo val/test. Bốn biến thể đồng thời thay nhiều bước nên chưa tách riêng tác động nhân quả của từng phép FE.",
        "02_benchmark.ipynb": f"## Phân tích benchmark đã hoàn thiện\n\nRF Original được chọn trước test bằng CV accuracy {chosen.cv_mean:.4f} ± {chosen.cv_std:.4f}. Test có TN={tn}, FP={fp}, FN={fn}, TP={tp}; accuracy {chosen.test_accuracy:.4f}, recall lớp 1 {chosen.test_recall:.4f}. Một false negative ở dòng CSV 67 có xác suất 0,3862; không đổi ngưỡng theo lỗi này. FE chỉ tăng CV ở {fe_improved}/7 mô hình, Original+DT ở {dt_improved}/7 nên không có cải thiện nhất quán. Một mẫu test làm accuracy đổi 3,23 điểm phần trăm. Repeated CV F1 trên train {stability['mean']:.4f} ± {stability['std']:.4f} là độ ổn định mô tả, không thay nested CV. KMeans/stacking nằm trong bảng riêng, thêm sau khi đã xem test, không dùng để chọn lại demo.",
        "03_treeshap.ipynb": f"## Diễn giải SHAP đã hoàn thiện\n\nTrên 31 mẫu, cp/thal/ca có mean |SHAP| lớn nhất, tiếp theo oldpeak/thalach. Mẫu local index {analysis['shap_local_index']} được chọn vì là true positive có xác suất cao nhất, không đại diện ngẫu nhiên. Giá trị nền {analysis['shap_base_value']:.4f} cộng đóng góp thành {shap_info['probability']:.4f}, khớp predict_proba với sai số tối đa {shap_info['max_additivity_error']:.2e}. thal/cp/ca/oldpeak đẩy đầu ra tăng trong ví dụ này. Feature hiển thị là sau preprocessing. Beeswarm mô tả liên hệ trong model và mẫu quan sát; không chứng minh nguyên nhân bệnh hoặc tác động điều trị.",
    }
    for name, narrative in narratives.items():
        path = root / "notebooks" / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        notebook["cells"][-1]["source"] = narrative.splitlines(keepends=True)
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Đã tạo final_report.md, analysis/error/demo/SHAP tables và nhận xét cho bốn notebook.")
    return reports / "final_report.md"

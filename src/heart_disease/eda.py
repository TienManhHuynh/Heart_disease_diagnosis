"""Thống kê và biểu đồ khám phá; không sửa dữ liệu đầu vào."""

from pathlib import Path

from .config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def summarize(frame) -> dict:
    return {
        "overview": {
            "rows": len(frame),
            "columns": frame.shape[1],
            "duplicate_rows": int(frame.duplicated().sum()),
        },
        "dtypes": frame.dtypes.astype(str).rename("dtype").to_frame(),
        "describe": frame.describe(include="all").T,
        "missing": frame.isna().sum().rename("missing").to_frame(),
        "target_counts": frame["target"].value_counts().sort_index().rename("count").to_frame(),
    }


def plot_eda(frame, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        path = output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for column, ax in zip(NUMERIC_COLUMNS, axes.flat):
        sns.histplot(frame[column], bins=15, ax=ax)
        ax.set_title(column)
    axes.flat[-1].set_visible(False)
    save(fig, "train_numeric.png")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for column, ax in zip(CATEGORICAL_COLUMNS, axes.flat):
        sns.countplot(data=frame, x=column, ax=ax)
        ax.set_title(column)
    save(fig, "train_categorical.png")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(frame.corr(numeric_only=True), cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation on training subset (numeric codes included)")
    save(fig, "train_correlation.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    frame.isna().sum().plot.bar(ax=axes[0], title="Missing values on train")
    frame["target"].value_counts().sort_index().plot.bar(
        ax=axes[1], title="Training target distribution"
    )
    save(fig, "train_missing_target.png")
    return paths

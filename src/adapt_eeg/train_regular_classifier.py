from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import Xdawn
from scipy.signal import resample
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import warnings

from adapt_eeg.data_reader import build_metadata, read_unepoch_sample

GroupBy = Literal["participant", "line", "participant_line"]


@dataclass(frozen=True)
class PreparedData:
    X: np.ndarray
    y: np.ndarray
    outer_groups: np.ndarray
    inner_groups: np.ndarray
    feature_names: list[str]
    samples: pd.DataFrame


def sample_to_array(data: mne.BaseEpochs | mne.io.BaseRaw) -> tuple[np.ndarray, list[str], float]:
    if isinstance(data, mne.BaseEpochs):
        epoch_data = data.get_data(copy=True)
        if epoch_data.ndim != 3:
            raise ValueError(f"Expected Epochs data with 3 dimensions, got {epoch_data.shape}")
        array = epoch_data[0] if epoch_data.shape[0] == 1 else epoch_data.mean(axis=0)
        ch_names = list(data.ch_names)
        sfreq = float(data.info["sfreq"])
    elif isinstance(data, mne.io.BaseRaw):
        array = data.get_data()
        ch_names = list(data.ch_names)
        sfreq = float(data.info["sfreq"])
    else:
        raise TypeError(f"Unsupported MNE object: {type(data)!r}")

    array = np.asarray(array, dtype=float)
    array = array - np.nanmean(array, axis=-1, keepdims=True)
    return array, ch_names, sfreq


def pick_channels(array: np.ndarray, ch_names: list[str], n_channels: int) -> tuple[np.ndarray, list[str]]:
    if len(ch_names) < n_channels:
        raise ValueError(f"Need at least {n_channels} channels, got {len(ch_names)}")
    return array[:n_channels], ch_names[:n_channels]


def time_resample(array: np.ndarray, n_times: int) -> np.ndarray:
    if array.shape[-1] == n_times:
        return array
    return resample(array, n_times, axis=-1)


def add_2hz_sinusoid_component(array: np.ndarray, sfreq: float) -> np.ndarray:
    n_times = array.shape[-1]
    t = np.arange(n_times, dtype=float) / sfreq
    sin_basis = np.sin(2 * np.pi * 2.0 * t)
    cos_basis = np.cos(2 * np.pi * 2.0 * t)
    design = np.column_stack([sin_basis, cos_basis])
    coef = np.linalg.pinv(design) @ array.T
    fitted = (design @ coef).T
    return array + fitted


def flatten_features(array: np.ndarray, ch_names: list[str]) -> tuple[np.ndarray, list[str]]:
    names = [f"{ch}_t{idx}" for ch in ch_names for idx in range(array.shape[-1])]
    return array.reshape(-1), names


def group_values(samples: pd.DataFrame, group_by: GroupBy) -> np.ndarray:
    if group_by == "participant":
        return samples["participant"].to_numpy()
    if group_by == "line":
        return samples["line"].to_numpy()
    if group_by == "participant_line":
        return (samples["participant"].astype(str) + "_" + samples["line"].astype(str)).to_numpy()
    raise ValueError(f"Unknown group_by={group_by!r}")


def prepare_data(
    data_root: Path,
    *,
    n_channels: int = 12,
    n_times: int = 20,
    use_2hz_component: bool = True,
    outer_group_by: GroupBy = "participant",
    inner_group_by: GroupBy = "participant",
) -> PreparedData:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    records: list[dict[str, int | str]] = []
    skipped: list[dict[str, int | str]] = []
    feature_names: list[str] | None = None

    for metadata in build_metadata(data_root):
        sample = read_unepoch_sample(metadata)
        array, ch_names, sfreq = sample_to_array(sample.data)
        if len(ch_names) < n_channels:
            skipped.append(
                {
                    "participant": int(sample.participant),
                    "line": int(sample.line),
                    "rhythm_type": sample.rhythm_type.value,
                    "reason": f"expected at least {n_channels} channels, got {len(ch_names)}",
                }
            )
            continue
        array, ch_names = pick_channels(array, ch_names, n_channels)
        if use_2hz_component:
            array = add_2hz_sinusoid_component(array, sfreq)
        array = time_resample(array, n_times)
        features, names = flatten_features(array, ch_names)

        if feature_names is None:
            feature_names = names

        rows.append(features)
        labels.append(sample.rhythm_type.value)
        records.append(
            {
                "participant": int(sample.participant),
                "line": int(sample.line),
                "rhythm_type": sample.rhythm_type.value,
                "language_understanding": sample.language_understanding.value,
            }
        )

    samples = pd.DataFrame(records)
    samples.attrs["skipped"] = pd.DataFrame(skipped)
    y = LabelEncoder().fit_transform(labels)
    return PreparedData(
        X=np.vstack(rows),
        y=y,
        outer_groups=group_values(samples, outer_group_by),
        inner_groups=group_values(samples, inner_group_by),
        feature_names=feature_names or [],
        samples=samples,
    )


class XdawnVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_channels: int = 12, n_times: int = 20, n_components: int = 2):
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray):
        epochs = self._as_epochs(X)
        self.xdawn_ = Xdawn(n_components=self.n_components, correct_overlap=False)
        self.xdawn_.fit(epochs, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        epochs = self._as_epochs(X)
        transformed = self.xdawn_.transform(epochs)
        if isinstance(transformed, dict):
            transformed = np.concatenate([transformed[key] for key in sorted(transformed)], axis=1)
        return np.asarray(transformed).reshape(len(X), -1)

    def _as_epochs(self, X: np.ndarray) -> mne.EpochsArray:
        data = np.asarray(X).reshape(len(X), self.n_channels, self.n_times)
        info = mne.create_info(
            ch_names=[f"EEG{idx:02d}" for idx in range(self.n_channels)],
            sfreq=float(self.n_times),
            ch_types="eeg",
        )
        events = np.column_stack(
            [np.arange(len(X)), np.zeros(len(X), dtype=int), np.ones(len(X), dtype=int)]
        )
        return mne.EpochsArray(data, info, events=events, tmin=0.0, baseline=None, verbose=False)


class EEGPTExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        if self.enabled:
            raise NotImplementedError(
                "EEGPT extraction is optional and not bundled. Add a project-specific "
                "EEGPT embedding extractor here, then keep it inside the nested-CV pipeline."
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


def make_pipelines(n_channels: int, n_times: int) -> dict[str, tuple[Pipeline, dict[str, list]]]:
    return {
        "raw_l2_logistic": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
                ]
            ),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "raw_shrinkage_lda": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", LinearDiscriminantAnalysis(solver="lsqr")),
                ]
            ),
            {"classifier__shrinkage": ["auto", 0.1, 0.5, 0.9]},
        ),
        "raw_linear_svm": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", LinearSVC(class_weight="balanced", dual="auto", max_iter=10000)),
                ]
            ),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "pca_l2_logistic": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                    ("pca", PCA()),
                    ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
                ]
            ),
            {
                "pca__n_components": [0.8, 0.9, 0.95],
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        "xdawn_shrinkage_lda": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("xdawn", XdawnVectorizer(n_channels=n_channels, n_times=n_times)),
                    ("scaler", StandardScaler()),
                    ("classifier", LinearDiscriminantAnalysis(solver="lsqr")),
                ]
            ),
            {
                "xdawn__n_components": [1, 2, 3],
                "classifier__shrinkage": ["auto", 0.1, 0.5, 0.9],
            },
        ),
        "xdawn_l2_logistic": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("xdawn", XdawnVectorizer(n_channels=n_channels, n_times=n_times)),
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
                ]
            ),
            {
                "xdawn__n_components": [1, 2, 3],
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
    }


def make_group_cv(n_splits: int, y: np.ndarray, groups: np.ndarray):
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two unique groups for grouped cross-validation")
    try:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    except Exception:
        return GroupKFold(n_splits=n_splits)


def decision_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def nested_cv(
    data: PreparedData,
    *,
    n_channels: int,
    n_times: int,
    outer_splits: int = 5,
    inner_splits: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_cv = make_group_cv(outer_splits, data.y, data.outer_groups)
    pipelines = make_pipelines(n_channels, n_times)
    fold_rows: list[dict] = []
    best_rows: list[dict] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for outer_fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(data.X, data.y, data.outer_groups), start=1
        ):
            X_train, X_test = data.X[train_idx], data.X[test_idx]
            y_train, y_test = data.y[train_idx], data.y[test_idx]
            inner_groups_train = data.inner_groups[train_idx]
            inner_cv = make_group_cv(inner_splits, y_train, inner_groups_train)

            for name, (pipeline, param_grid) in pipelines.items():
                search = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    scoring="balanced_accuracy",
                    cv=inner_cv,
                    n_jobs=-1,
                    refit=True,
                    error_score=np.nan,
                )
                search.fit(X_train, y_train, groups=inner_groups_train)
                pred = search.predict(X_test)
                score = decision_scores(search.best_estimator_, X_test)

                fold_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "model": name,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                        "f1_regular": f1_score(y_test, pred, pos_label=1),
                        "roc_auc_regular": roc_auc_score(y_test, score) if len(np.unique(y_test)) == 2 else np.nan,
                        "best_inner_balanced_accuracy": search.best_score_,
                    }
                )
                best_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "model": name,
                        **search.best_params_,
                    }
                )

    scores = pd.DataFrame(fold_rows)
    params = pd.DataFrame(best_rows)
    return scores, params


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.groupby("model").agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        f1_regular_mean=("f1_regular", "mean"),
        f1_regular_sd=("f1_regular", "std"),
        roc_auc_regular_mean=("roc_auc_regular", "mean"),
        roc_auc_regular_sd=("roc_auc_regular", "std"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train regular-vs-irregular EEG classifiers.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/itpc-30event-good"))
    parser.add_argument("--n-channels", type=int, default=12)
    parser.add_argument("--n-times", type=int, default=20)
    parser.add_argument("--outer-group-by", choices=["participant", "line", "participant_line"], default="participant")
    parser.add_argument("--inner-group-by", choices=["participant", "line", "participant_line"], default="participant")
    parser.add_argument("--without-2hz-component", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_training"))
    args = parser.parse_args()

    mne.set_log_level("ERROR")
    data = prepare_data(
        args.data_root,
        n_channels=args.n_channels,
        n_times=args.n_times,
        use_2hz_component=not args.without_2hz_component,
        outer_group_by=args.outer_group_by,
        inner_group_by=args.inner_group_by,
    )
    scores, params = nested_cv(data, n_channels=args.n_channels, n_times=args.n_times)
    summary = summarize_scores(scores)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "nested_cv_scores.csv", index=False)
    params.to_csv(args.output_dir / "nested_cv_best_params.csv", index=False)
    summary.to_csv(args.output_dir / "nested_cv_summary.csv")
    data.samples.to_csv(args.output_dir / "samples.csv", index=False)
    skipped = data.samples.attrs.get("skipped")
    if isinstance(skipped, pd.DataFrame) and not skipped.empty:
        skipped.to_csv(args.output_dir / "skipped_samples.csv", index=False)

    print(f"Samples: {len(data.samples)}")
    print(f"Outer groups: {len(np.unique(data.outer_groups))}")
    print(f"Inner groups: {len(np.unique(data.inner_groups))}")
    print(summary)


if __name__ == "__main__":
    main()

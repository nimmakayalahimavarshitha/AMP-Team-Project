
from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "GenPept_Curated_2025_primary_split_v1.1.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def amino_acid_composition(sequence):
    """Convert a protein sequence into 20 amino-acid percentages."""
    sequence = str(sequence).strip().upper()
    counts = {aa: 0 for aa in AMINO_ACIDS}

    for aa in sequence:
        if aa in counts:
            counts[aa] += 1

    valid_count = sum(counts.values())

    if valid_count == 0:
        raise ValueError("A sequence contains no standard amino acids.")

    return [counts[aa] / valid_count for aa in AMINO_ACIDS]


def calculate_metrics(y_true, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def main():
    print("Loading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Check that the CSV is inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"label", "split"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing required columns: {required_columns - set(df.columns)}"
        )

    sequence_column = (
        "sequence_clean" if "sequence_clean" in df.columns else "sequence"
    )

    if sequence_column not in df.columns:
        raise ValueError("No sequence or sequence_clean column was found.")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    label_map = {"amp": 1, "non-amp": 0}
    if not set(df["label"].dropna().unique()).issubset(label_map):
        raise ValueError(
            f"Unexpected labels found: {df['label'].unique().tolist()}"
        )

    df["target"] = df["label"].map(label_map)

    print("\nDataset shape:", df.shape)
    print("\nLabel counts:")
    print(df["label"].value_counts())
    print("\nSplit counts:")
    print(df["split"].value_counts())
    print("\nLabels by split:")
    print(pd.crosstab(df["split"], df["label"]))

    # Convert sequences into amino-acid composition features.
    print("\nExtracting amino-acid composition features...")

    df = df.dropna(subset=[sequence_column, "target", "split"]).copy()
    df[sequence_column] = df[sequence_column].astype(str).str.strip().str.upper()

    feature_rows = []
    valid_indices = []

    for idx, sequence in zip(df.index, df[sequence_column]):
        try:
            feature_rows.append(amino_acid_composition(sequence))
            valid_indices.append(idx)
        except ValueError:
            print(f"Warning: skipping sequence at row index {idx}")

    df = df.loc[valid_indices].copy()
    X = pd.DataFrame(
        feature_rows,
        index=df.index,
        columns=[f"AAC_{aa}" for aa in AMINO_ACIDS],
    )

    print("Number of usable sequences:", len(df))
    print("Number of features per sequence:", X.shape[1])

    train_mask = df["split"] == "train"
    val_mask = df["split"].isin(["val", "validation"])
    test_mask = df["split"] == "test"

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(
            "Could not find all train, val/validation, and test splits."
        )

    X_train = X.loc[train_mask]
    y_train = df.loc[train_mask, "target"].astype(int)

    X_val = X.loc[val_mask]
    y_val = df.loc[val_mask, "target"].astype(int)

    X_test = X.loc[test_mask]
    y_test = df.loc[test_mask, "target"].astype(int)

    print("\nTraining samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Test samples:", len(X_test))

    # Tune parameters using validation data only.
    print("\nSelecting Random Forest parameters using validation data...")

    parameter_options = itertools.product(
        [200, 400],       # Number of trees
        [None, 20],       # Maximum tree depth
        [1, 2],            # Minimum samples per leaf
    )

    tuning_results = []
    best_params = None
    best_f1 = -1.0

    for n_estimators, max_depth, min_samples_leaf in parameter_options:
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        val_predictions = model.predict(X_val)
        val_probabilities = model.predict_proba(X_val)[:, 1]

        val_metrics = calculate_metrics(
            y_val, val_predictions, val_probabilities
        )

        result = {
            **params,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        tuning_results.append(result)

        print(
            f"Trees={n_estimators}, depth={max_depth}, "
            f"leaf={min_samples_leaf}, "
            f"validation F1={val_metrics['f1_score']:.4f}"
        )

        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            best_params = params

    pd.DataFrame(tuning_results).to_csv(
        RESULTS_DIR / "member2_rf_tuning.csv", index=False
    )

    print("\nBest parameters:", best_params)
    print(f"Best validation F1-score: {best_f1:.4f}")

    # Refit on train + validation using the selected parameters.
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    final_model = RandomForestClassifier(**best_params)
    final_model.fit(X_train_val, y_train_val)

    # Evaluate the held-out test split only after parameter selection.
    test_predictions = final_model.predict(X_test)
    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    test_metrics = calculate_metrics(
        y_test, test_predictions, test_probabilities
    )

    print("\n========== FINAL TEST RESULTS ==========")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["Non-AMP", "AMP"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    # Save metrics.
    output_metrics = {
        "model": "Random Forest",
        "feature_method": "Amino Acid Composition (AAC), 20 features",
        "selected_parameters": {
            key: value for key, value in best_params.items()
            if key not in ["n_jobs", "random_state"]
        },
        "best_validation_f1": float(best_f1),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "test_metrics": test_metrics,
        "confusion_matrix_labels": ["Non-AMP", "AMP"],
        "confusion_matrix": matrix.tolist(),
    }

    with open(
        RESULTS_DIR / "member2_rf_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(output_metrics, file, indent=4)

    # Save individual test predictions.
    predictions_df = pd.DataFrame({
        "row_id": df.loc[X_test.index, "row_id"]
        if "row_id" in df.columns
        else X_test.index,
        "actual_label": y_test.map({0: "Non-AMP", 1: "AMP"}).values,
        "predicted_label": pd.Series(
            test_predictions, index=X_test.index
        ).map({0: "Non-AMP", 1: "AMP"}).values,
        "AMP_probability": test_probabilities,
    })

    predictions_df.to_csv(
        RESULTS_DIR / "member2_rf_predictions.csv", index=False
    )

    # Plot feature importance.
    importances = pd.Series(
        final_model.feature_importances_,
        index=X_train.columns,
    ).sort_values()

    plt.figure(figsize=(9, 7))
    importances.plot(kind="barh")
    plt.xlabel("Feature importance")
    plt.ylabel("Amino-acid composition feature")
    plt.title("Random Forest: Amino-Acid Feature Importance")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "member2_rf_feature_importance.png",
        dpi=300,
    )
    plt.close()

    print("\nAll results saved in:", RESULTS_DIR)
    print("1. member2_rf_metrics.json")
    print("2. member2_rf_predictions.csv")
    print("3. member2_rf_tuning.csv")
    print("4. member2_rf_feature_importance.png")


if __name__ == "__main__":
    main()

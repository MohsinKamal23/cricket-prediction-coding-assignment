import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from typing import Tuple, Dict, Any, Optional
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, average_precision_score



def load_registry(model_dir: str) -> dict:
    """
    Load the model registry from a JSON file inside the given directory.

    The registry keeps track of all trained models, their parameters,
    and evaluation metrics.

    Args:
        model_dir (str): Path to the model directory.

    Returns:
        dict: Registry dictionary if found, otherwise an empty dict.
    """
    registry_file = os.path.join(model_dir, "registry.json")
    if os.path.exists(registry_file):
        with open(registry_file, "r") as f:
            return json.load(f)
    return {}

def save_registry(model_dir: str, registry: dict):
    """
    Save the model registry dictionary to a JSON file.

    Args:
        model_dir (str): Path to the model directory.
        registry (dict): Registry dictionary to save.

    Returns:
        None
    """
    registry_file = os.path.join(model_dir, "registry.json")
    with open(registry_file, "w") as f:
        json.dump(registry, f, indent=4)

def set_active_model(model_dir: str, model_name: str, model_file: str):
    """
    Set the currently active model by writing metadata into a JSON file.

    Args:
        model_dir (str): Path to the model directory.
        model_name (str): Name of the model type (e.g., Logistic, RandomForest).
        model_file (str): Model file name to set as active.

    Returns:
        None
    """
    active_file = os.path.join(model_dir, "active_model.json")
    active_info = {"model_name": model_name, "model_file": model_file}
    with open(active_file, "w") as f:
        json.dump(active_info, f, indent=4)


def perform_eda_and_visualizations( train_df: pd.DataFrame, model_dir: str, test_df: Optional[pd.DataFrame] = None ):    
    """
    Perform exploratory data analysis (EDA) and generate key visualizations.

    This function creates plots to analyze:
      - Class balance (win vs loss)
      - Correlation heatmap of numeric features
      - Distribution of balls left by outcome
      - Distribution of total runs by outcome

    The plots are saved inside a subdirectory `eda/` under the given model directory.

    Args:
        train_df (pd.DataFrame): Training dataset with features and target 'won'.
        model_dir (str): Directory where EDA plots will be saved.
    """
    eda_dir = os.path.join(model_dir, "eda")
    os.makedirs(eda_dir, exist_ok=True)

    print("\nClass balance in training dataset:")
    print(train_df['won'].value_counts())
    plt.figure(figsize=(6, 4))
    sns.countplot(x='won', data=train_df)
    plt.title("Win vs Loss Distribution")
    plt.savefig(os.path.join(eda_dir, "class_balance.png"))
    plt.close()

    # --- Correlation Heatmap ---
    plt.figure(figsize=(8, 6))
    corr = train_df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(eda_dir, "correlation_heatmap.png"))
    plt.close()

    # --- Balls left distribution ---
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_df, x="balls_left", hue="won", kde=True, element="step", stat="density", common_norm=False)
    plt.title("Distribution of Balls Left by Outcome")
    plt.savefig(os.path.join(eda_dir, "balls_left_distribution.png"))
    plt.close()

    # --- Total runs distribution ---
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_df, x="total_runs", hue="won", kde=True, element="step", stat="density", common_norm=False)
    plt.title("Distribution of Total Runs by Outcome")
    plt.savefig(os.path.join(eda_dir, "total_runs_distribution.png"))
    plt.close()

    plt.close()


def preprocess_cricket_dataset(filepath: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and preprocess the cricket dataset.

    - Drops missing values
    - Creates a `match_id` column by detecting innings resets
    - Identifies matches with inconsistent labels

    Args:
        filepath (str): Path to the cricket dataset CSV.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - Processed DataFrame
            - Summary dictionary with metadata about matches
    """
    df = pd.read_csv(filepath).dropna()
    df["match_id"] = (df["balls_left"].diff().fillna(5) > 4).cumsum()
    num_matches = df["match_id"].nunique()
    won_check = df.groupby("match_id")["won"].nunique()
    inconsistent_match_ids = won_check[won_check > 1].index.tolist()
    summary = {
        "num_matches": num_matches,
        "num_inconsistent": len(inconsistent_match_ids),
        "inconsistent_match_ids": inconsistent_match_ids,
    }
    return df, summary

def split_train_val(df: pd.DataFrame, feature_cols: list,
                    test_size: float = 0.2, random_state: int = 42):
    """
    Perform match-based train-validation split to avoid data leakage.

    Splits entire matches into training or validation sets
    instead of splitting individual rows.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        feature_cols (list): List of feature column names.
        test_size (float): Fraction of matches for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple:
            - X_train (pd.DataFrame)
            - X_val (pd.DataFrame)
            - y_train (pd.Series)
            - y_val (pd.Series)
            - split_info (dict): Summary of split sizes
    """
    unique_matches = df["match_id"].unique()
    match_labels = df.groupby("match_id")["won"].first()
    train_matches, val_matches = train_test_split(
        unique_matches,
        test_size=test_size,
        random_state=random_state,
        stratify=match_labels.loc[unique_matches]
    )
    X_train = df[df["match_id"].isin(train_matches)][feature_cols]
    y_train = df[df["match_id"].isin(train_matches)]["won"]
    X_val = df[df["match_id"].isin(val_matches)][feature_cols]
    y_val = df[df["match_id"].isin(val_matches)]["won"]
    split_info = {
        "num_train_matches": len(train_matches),
        "num_val_matches": len(val_matches),
    }
    return X_train, X_val, y_train, y_val, split_info


def define_models_and_params() -> Dict[str, Any]:
    """
    Define candidate models and their hyperparameter grids.

    Returns:
        Dict[str, Any]: Dictionary mapping model names to
                        (estimator, hyperparameter grid).
    """
    return {
        "Logistic": (
            LogisticRegression(max_iter=1000),
            {"C": [0.1, 1, 10], "solver": ["liblinear", "lbfgs"], "class_weight": ["balanced"]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [3, 5, 10], "min_samples_split": [2, 5], "class_weight": ["balanced"]}
        ),
        "XGBoost": (
    XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ),
    {
        "n_estimators": [100, 200, 400],          
        "max_depth": [3, 5, 7],              
        "learning_rate": [0.01, 0.05],  
        "subsample": [0.8, 1.0],            
        "colsample_bytree": [0.8, 1.0],    
        "scale_pos_weight": [0.6]      
    })
    }


def train_and_evaluate(data_path: str, model_dir: str):
    """
    Train and evaluate multiple ML models for cricket outcome prediction.

    - Loads and preprocesses dataset
    - Performs EDA and saves plots
    - Splits dataset into train/validation sets
    - Runs GridSearchCV for Logistic, RandomForest, and XGBoost
    - Evaluates using F1, Accuracy, and Average Precision
    - Saves confusion matrices, classification reports, and predictions
    - Versions models in a registry and sets the best model active

    Args:
        data_path (str): Path to the cricket dataset CSV file.
        model_dir (str): Directory to save models, registry, and outputs.

    Returns:
        dict: Evaluation results for all models.
    """
    os.makedirs(model_dir, exist_ok=True)
    registry = load_registry(model_dir)

    df, summary = preprocess_cricket_dataset(data_path)
    print("Dataset summary:", summary)

    perform_eda_and_visualizations(df, model_dir)

    feature_cols = ["total_runs", "wickets", "balls_left", "target"]
    X_train, X_val, y_train, y_val, split_info = split_train_val(df, feature_cols)
    print("Split info:", split_info)

    results = {}
    best_model_name, best_f1, best_model_file = None, -1, None
    models_and_params = define_models_and_params()

    for name, (estimator, param_grid) in models_and_params.items():
        print(f"Running GridSearchCV for {name}...")
        grid = GridSearchCV(estimator, param_grid, scoring="f1", cv=5, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_val)
        y_proba = best_model.predict_proba(X_val)[:, 1]

        results[name] = {
            "best_params": grid.best_params_,
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "auc_pr": average_precision_score(y_val, y_proba)
        }
        
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"{name} - Confusion Matrix")
        plt.savefig(os.path.join(model_dir, f"{name.lower()}_confusion_matrix.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n{name} Confusion Matrix:\n", cm)
        
        print(f"\n{name} Classification Report:")
        print(classification_report(y_val, y_pred, digits=3))

        version_number = sum(1 for k in registry if k.startswith(name.lower())) + 1
        model_filename = f"{name.lower()}_v{version_number}.pkl"
        model_path = os.path.join(model_dir, model_filename)
        joblib.dump(best_model, model_path)

        registry[name.lower()] = {
        "latest": model_filename,
        "metrics": {
            "accuracy": results[name]["accuracy"],
            "f1": results[name]["f1"],
            "auc_pr": results[name]["auc_pr"]
        },
        "best_params": results[name]["best_params"]
}

        if results[name]["auc_pr"] > best_f1:
            best_f1 = results[name]["auc_pr"]
            best_model_name, best_model_file = name, model_filename

        # Save validation predictions
        val_preds = pd.DataFrame(X_val.copy())
        val_preds["true_label"] = y_val.values
        val_preds["predicted_label"] = y_pred
        val_preds["confidence"] = y_proba
        val_preds.to_csv(os.path.join(model_dir, f"{name.lower()}_val_predictions.csv"), index=False)

    save_registry(model_dir, registry)
    if best_model_name and best_model_file:
        set_active_model(model_dir, best_model_name, best_model_file)

    print("Final Results:", results)
    return results

# ----------------------------
# Run Training
# ----------------------------
if __name__ == "__main__":
    data_path = "cricket_dataset.csv"
    model_dir = "models"
    train_and_evaluate(data_path, model_dir)
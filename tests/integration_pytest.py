# test_pipeline.py
import pandas as pd
import pytest
from train import train_and_evaluate


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a synthetic cricket dataset with multiple matches for testing."""
    num_matches = 6
    rows_per_match = 10

    data = {
        "match_id": [i for i in range(num_matches) for _ in range(rows_per_match)],
        "balls_left": [120 - j*5 for i in range(num_matches) for j in range(rows_per_match)],
        "total_runs": [j*10 for i in range(num_matches) for j in range(rows_per_match)],
        "wickets":   [j//2 for i in range(num_matches) for j in range(rows_per_match)],
        "target":    [250] * (num_matches * rows_per_match),
        "won":       [0, 1] * (num_matches * rows_per_match // 2),
    }

    df = pd.DataFrame(data)
    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_integration_train_and_evaluate(sample_dataset, tmp_path):
    """Integration test: full pipeline end-to-end."""
    model_dir = tmp_path / "models"

    # Run training pipeline
    results = train_and_evaluate(sample_dataset, str(model_dir))

    # ---- Assertions ----
    # 1. Results dictionary is returned
    assert isinstance(results, dict)
    assert "Logistic" in results
    assert "RandomForest" in results
    assert "XGBoost" in results

    # 2. Metrics are within [0,1]
    for model, metrics in results.items():
        for metric_name in ["accuracy", "f1", "auc_pr"]:
            assert 0.0 <= metrics[metric_name] <= 1.0

    # 3. Registry + active model saved
    assert (model_dir / "registry.json").exists()
    assert (model_dir / "active_model.json").exists()

    # 4. At least one trained model saved
    saved_models = list(model_dir.glob("*.pkl"))
    assert len(saved_models) > 0

    # 5. Validation predictions CSVs saved
    saved_preds = list(model_dir.glob("*_val_predictions.csv"))
    assert len(saved_preds) == 3   # one per model
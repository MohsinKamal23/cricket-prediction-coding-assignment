
Cricket Match Prediction API
A FastAPI-based machine learning project that predicts the outcome of cricket matches based on ball-by-ball data.
It supports predictions via REST API, along with explanations powered by Google Gemini LLM.

Features
EDA visualizations (class balance, distributions, correlations)
Train ML models (Logistic Regression, Random Forest, XGBoost, SVM) on cricket datasets
Match-based train-test split to avoid data leakage
Model versioning with active_model.json
API endpoints for predictions (/predict) and explanations (/explain/{id})
Integration tests with pytest

Setup Instructions
1. Clone the Repository
git clone https://github.com/MohsinKamal23/cricket-prediction-coding-assignment.git
cd cricket-prediction

2. Install Dependencies
pip install -r requirements.txt

3. Set Environment Variables
Set your Google Gemini API Key:
set GOOGLE_API_KEY=your_api_key_here   

4. Train Model
python train.py
This will train models, save them in models/, and mark the best one as active_model.json.

5. Run FastAPI Server
uvicorn app:app --reload
API will be available at: http://127.0.0.1:8000/docs (interactive Swagger UI)

6. Predict Match Outcome
Upload CSV with required columns (total_runs, wickets, balls_left, target):
curl -X POST "http://127.0.0.1:8000/predict" -F "file=cricket_dataset_test.csv"
Response:
{
  "status": "success",
  "predictions_file": "models/predictions/results.csv",
  "metadata": {
    "total_rows": 627,
    "filtered_rows": 185,
    "predictions_made": 185,
    "model_used": "randomforest_v1.pkl"
  }
}

7. Explain Prediction
curl -X POST "http://127.0.0.1:8000/explain/0"
Response:
{
  "prediction": 1,
  "confidence": 0.82,
  "explanation": "The batting side has scored consistently and only lost 2 wickets..."
}

8. Testing
Run integration tests:
pytest -v integration_pytest.py

9. Model Performance Summary
Model	            Accuracy  F1 Score	AUC-PR
Logistic Regression	0.84	  0.88	    0.89
Random Forest	    0.86	  0.90	    0.95
XGBoost	0.84	    0.85	  0.88      0.89
Final active model: Random Forest (best_model.pkl)

📂 Project Structure
cricket-prediction/
│── app.py                   # FastAPI app with prediction + explanation endpoints
│── train.py                 # Training pipeline (model training & evaluation)
│── eda/                     # Saved visualizations (EDA outputs)
│── data/                     # Raw datasets
│   ├── cricket_dataset.csv             # Training dataset
│   └── cricket_dataset_test.csv              # Test dataset
│── tests/
│   └── integration_pytest.py # Integration tests using pytest
│── predictions/             # Generated prediction CSV
│── models/                  # Saved models, registry.json, active_model.json
│── requirements.txt         # Python dependencies
│── Project Report.docx      # Project report (documentation)
│── README.md                # Setup guide, usage, and summary
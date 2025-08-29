import os
import io
import json
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Path
from fastapi.responses import JSONResponse

# Google Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------
# Config
# ----------------------------
MODEL_DIR = "models"
PREDICTION_DIR = os.path.join(MODEL_DIR, "predictions")
os.makedirs(PREDICTION_DIR, exist_ok=True)

app = FastAPI(title="Cricket Prediction API", version="1.0")

# Setup Google Gemini
os.environ["GOOGLE_API_KEY"] = ""  # 🔑 set your key here
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    max_output_tokens=512
)

# ----------------------------
# Helpers
# ----------------------------
def load_active_model(model_dir: str):
    """
    Load the currently active trained model.

    Args:
        model_dir (str): Path to the directory where models are stored.

    Returns:
        tuple: (model, active_info)
            - model: The loaded scikit-learn model.
            - active_info (dict): Metadata about the active model.

    Raises:
        FileNotFoundError: If no active_model.json or model file is found.
    """
    active_file = os.path.join(model_dir, "active_model.json")
    if not os.path.exists(active_file):
        raise FileNotFoundError("Active model not set. Please train a model first.")

    with open(active_file, "r") as f:
        active_info = json.load(f)

    model_path = os.path.join(model_dir, active_info["model_file"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model, active_info

def validate_csv(df: pd.DataFrame):
    """
    Validate that the uploaded CSV has the required columns.

    Args:
        df (pd.DataFrame): Input dataframe from uploaded CSV.

    Raises:
        ValueError: If any required column is missing.
    """
    required_cols = ["total_runs", "wickets", "balls_left", "target"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/")
def home():
    """
    Root endpoint to verify that the API is running.

    Returns:
        dict: Welcome message.
    """
    return {"msg": "Cricket Prediction API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a CSV file and generate match outcome predictions.

    Steps:
        1. Reads uploaded CSV into a DataFrame.
        2. Validates required columns.
        3. Applies filtering: `balls_left < 60` and `target > 120`.
        4. Runs the active ML model to predict outcomes.
        5. Saves results to `results.csv` in prediction directory.

    Args:
        file (UploadFile): CSV file containing match features.

    Returns:
        JSONResponse: Metadata and path to saved predictions.

    Raises:
        HTTPException: If validation fails or prediction cannot be made.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        validate_csv(df)

        filtered_df = df[(df["balls_left"] < 60) & (df["target"] > 120)].copy()
        if filtered_df.empty:
            raise ValueError("No rows matched filter condition: balls_left < 60 and target > 120")

        model, active_info = load_active_model(MODEL_DIR)
        X = filtered_df[["total_runs", "wickets", "balls_left", "target"]]

        # Predictions + confidence
        filtered_df["prediction"] = model.predict(X)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            # pick probability of predicted class
            filtered_df["confidence"] = [
                probs[i, pred] for i, pred in enumerate(filtered_df["prediction"])
            ]
        else:
            filtered_df["confidence"] = 0.5  # fallback

        # Save to CSV
        pred_file_path = os.path.join(PREDICTION_DIR, "results.csv")
        filtered_df.to_csv(pred_file_path, index=False)

        return JSONResponse(content={
            "status": "success",
            "predictions_file": pred_file_path,
            "metadata": {
                "total_rows": len(df),
                "filtered_rows": len(filtered_df),
                "predictions_made": len(filtered_df),
                "model_used": active_info["model_file"]
            }
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain/{prediction_id}")
async def explain_prediction(prediction_id: int = Path(..., description="Row index of prediction in results.csv")):
    """
    Explain a specific prediction using Google Gemini LLM.

    Args:
        prediction_id (int): Row index of the prediction in results.csv.

    Returns:
        dict: Prediction, confidence, and AI-generated explanation.

    Raises:
        HTTPException: If predictions file does not exist or ID is invalid.
    """
    try:
        # Load results.csv
        results_path = os.path.join(PREDICTION_DIR, "results.csv")
        if not os.path.exists(results_path):
            raise FileNotFoundError("No predictions found. Please run /predict first.")
        
        df = pd.read_csv(results_path)

        if prediction_id >= len(df):
            raise ValueError(f"Prediction ID {prediction_id} not found in results.csv")

        row = df.iloc[prediction_id]

        # Prompt (using only your required fields)
        prompt = f"""
        You are a cricket match prediction assistant. 
        A machine learning model predicted the following:

        Prediction: {row['prediction']}
        Confidence: {row['confidence']:.2f}

        Match context:
        - Runs Scored: {row['total_runs']}
        - Wickets Lost: {row['wickets']}
        - Target: {row['target']}
        - Balls Left: {row['balls_left']}

        Explain the prediction in clear, human terms:
        - If confidence is high (>0.65), emphasize certainty and key drivers.
        - If confidence is medium (0.5-0.65), explain uncertainty and influencing factors.
        - If confidence is low (<0.5), emphasize unpredictability and risk factors.
        Make it concise but insightful (max 5 sentences).
        """

        response = llm.invoke(prompt)

        return {
            "prediction": int(row["prediction"]),
            "confidence": float(row["confidence"]),
            "explanation": response.content.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
import os
import tempfile
import pandas as pd
import joblib
from flask import Flask, request, render_template, send_file, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "churnsense_secret_key"

# Paths
MODEL_PATH = "model.pkl"
COLUMNS_PATH = "model_columns.pkl"
PREDICTIONS_OUTPUT = os.path.join(tempfile.gettempdir(), "predictions_output.csv")

model = None
model_columns = None
model_loaded = False

def load_model():
    global model, model_columns, model_loaded
    if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            model_columns = joblib.load(COLUMNS_PATH)
            model_loaded = True
            print("Model and columns loaded successfully.")
        except Exception as e:
            print(f"Error loading model files: {e}")
            model_loaded = False
    else:
        print("Model or columns file not found. Please run churn_model.py first.")
        model_loaded = False

# Initial attempt to load model
load_model()

@app.route("/", methods=["GET"])
def index():
    # Attempt reload if not loaded yet
    if not model_loaded:
        load_model()
    
    return render_template("index.html", model_loaded=model_loaded)

@app.route("/predict", methods=["POST"])
def predict():
    global model, model_columns, model_loaded
    if not model_loaded:
        load_model()
        if not model_loaded:
            return render_template("index.html", model_loaded=False, error_msg="Model files not found. Please run 'python churn_model.py' to train the model first.")

    try:
        # Get all 19 form fields
        form_data = {
            "gender": request.form.get("gender"),
            "SeniorCitizen": int(request.form.get("SeniorCitizen", 0)),
            "Partner": request.form.get("Partner"),
            "Dependents": request.form.get("Dependents"),
            "tenure": int(request.form.get("tenure", 0)),
            "PhoneService": request.form.get("PhoneService"),
            "MultipleLines": request.form.get("MultipleLines"),
            "InternetService": request.form.get("InternetService"),
            "OnlineSecurity": request.form.get("OnlineSecurity"),
            "OnlineBackup": request.form.get("OnlineBackup"),
            "DeviceProtection": request.form.get("DeviceProtection"),
            "TechSupport": request.form.get("TechSupport"),
            "StreamingTV": request.form.get("StreamingTV"),
            "StreamingMovies": request.form.get("StreamingMovies"),
            "Contract": request.form.get("Contract"),
            "PaperlessBilling": request.form.get("PaperlessBilling"),
            "PaymentMethod": request.form.get("PaymentMethod"),
            "MonthlyCharges": float(request.form.get("MonthlyCharges", 0.0)),
            "TotalCharges": float(request.form.get("TotalCharges", 0.0))
        }

        # Build single-row DataFrame
        df_input = pd.DataFrame([form_data])

        # Apply get_dummies
        df_input_encoded = pd.get_dummies(df_input, drop_first=False)

        # Align columns to match training set
        df_input_reindexed = df_input_encoded.reindex(columns=model_columns, fill_value=0)
        df_input_reindexed = df_input_reindexed.astype({col: int for col in df_input_reindexed.select_dtypes(include=['bool']).columns})

        # Run predictions
        pred = model.predict(df_input_reindexed)[0]
        prob = model.predict_proba(df_input_reindexed)[0][1]

        # Formatting outputs
        result = "CHURN" if pred == 1 else "STAY"
        probability_pct = f"{prob * 100:.1f}%"
        color = "red" if pred == 1 else "green"

        # Calculate top 5 feature importances
        importances = model.feature_importances_
        feature_names = model_columns
        sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        top_features = [{"name": name, "importance": round(float(imp) * 100, 2)} for name, imp in sorted_features[:5]]

        # Generate custom reasoning based on risk/loyalty features
        reasons = []
        if pred == 1:
            if form_data.get("Contract") == "Month-to-month":
                reasons.append("a month-to-month contract (lack of contract stability)")
            if form_data.get("tenure") <= 12:
                reasons.append(f"short relationship tenure ({form_data.get('tenure')} months)")
            if form_data.get("InternetService") == "Fiber optic":
                reasons.append("fiber optic internet connection (historically higher churn rates)")
            if form_data.get("OnlineSecurity") == "No":
                reasons.append("lack of Online Security services")
            if form_data.get("TechSupport") == "No":
                reasons.append("lack of Tech Support services")
            if form_data.get("PaymentMethod") == "Electronic check":
                reasons.append("payment method set to Electronic Check")
            if form_data.get("MonthlyCharges") > 70:
                reasons.append(f"higher-than-average Monthly Charges (${form_data.get('MonthlyCharges')})")
            
            if reasons:
                reason = "This customer is flagged for Churn risk primarily due to: " + ", ".join(reasons[:3]) + "."
            else:
                reason = "Based on demographic and service details, this customer exhibits a combination of features correlating with high customer attrition."
        else:
            if form_data.get("Contract") in ["One year", "Two year"]:
                reasons.append(f"a long-term contract ({form_data.get('Contract')})")
            if form_data.get("tenure") > 24:
                reasons.append(f"established customer loyalty ({form_data.get('tenure')} months tenure)")
            if form_data.get("OnlineSecurity") == "Yes":
                reasons.append("active Online Security protection")
            if form_data.get("TechSupport") == "Yes":
                reasons.append("access to dedicated Tech Support")
            if form_data.get("MonthlyCharges") <= 50:
                reasons.append(f"low monthly fee rate (${form_data.get('MonthlyCharges')})")
            
            if reasons:
                reason = "This customer is likely to Stay, supported by: " + ", ".join(reasons[:3]) + "."
            else:
                reason = "This customer profile suggests high stability with low attrition risk under their current service plan."

        return render_template(
            "index.html",
            model_loaded=True,
            result=result,
            probability=probability_pct,
            color=color,
            top_features=top_features,
            reason=reason,
            form_data=form_data
        )

    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template("index.html", model_loaded=True, error_msg=f"An error occurred during prediction: {str(e)}")

@app.route("/bulk_predict", methods=["GET", "POST"])
def bulk_predict():
    global model, model_columns, model_loaded
    if request.method == "GET":
        return render_template("bulk.html", predictions_exist=False)

    # POST - Handle file upload
    if not model_loaded:
        load_model()
        if not model_loaded:
            return render_template("bulk.html", error_msg="Model files not found. Please run 'python churn_model.py' to train the model first.")

    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("bulk.html", error_msg="No file selected. Please upload a valid CSV file.")

    try:
        # Read the uploaded CSV
        df_bulk = pd.read_csv(file)

        # Basic verification: make sure we have at least tenure or monthly charges
        required_minimum = ["tenure", "MonthlyCharges", "TotalCharges"]
        missing_minimum = [col for col in required_minimum if col not in df_bulk.columns]
        if missing_minimum:
            return render_template("bulk.html", error_msg=f"Invalid CSV format. Missing key columns: {', '.join(missing_minimum)}")

        # Keep output structure (we will append prediction results to the original input)
        df_output = df_bulk.copy()

        # Preprocess features
        df_model = df_bulk.copy()
        if "customerID" in df_model.columns:
            df_model = df_model.drop(columns=["customerID"])
        if "Churn" in df_model.columns:
            df_model = df_model.drop(columns=["Churn"])

        # Clean TotalCharges numeric column
        if "TotalCharges" in df_model.columns:
            df_model["TotalCharges"] = pd.to_numeric(df_model["TotalCharges"], errors="coerce").fillna(0.0)

        # Apply get_dummies
        df_model_encoded = pd.get_dummies(df_model, drop_first=False)

        # Align columns
        df_model_reindexed = df_model_encoded.reindex(columns=model_columns, fill_value=0)
        df_model_reindexed = df_model_reindexed.astype({col: int for col in df_model_reindexed.select_dtypes(include=['bool']).columns})

        # Run model predictions
        preds = model.predict(df_model_reindexed)
        probs = model.predict_proba(df_model_reindexed)[:, 1]

        # Append predictions to the output data
        df_output["Prediction"] = ["CHURN" if p == 1 else "STAY" for p in preds]
        df_output["Churn Probability (%)"] = [round(float(pr) * 100, 2) for pr in probs]

        # Save to predictions_output.csv for download (using temp directory for cloud compatibility)
        df_output.to_csv(PREDICTIONS_OUTPUT, index=False)

        # Calculate summary statistics
        total_rows = len(df_output)
        churn_count = int(sum(preds == 1))
        stay_count = int(sum(preds == 0))
        churn_rate = round((churn_count / total_rows) * 100, 1) if total_rows > 0 else 0.0

        # Convert table to dictionary for templates rendering
        # Limit preview to first 100 rows to prevent huge browser load times, but output CSV contains all
        table_preview = df_output.head(100).to_dict(orient="records")
        table_columns = df_output.columns.tolist()

        return render_template(
            "bulk.html",
            predictions_exist=True,
            total_rows=total_rows,
            churn_count=churn_count,
            stay_count=stay_count,
            churn_rate=churn_rate,
            table_preview=table_preview,
            table_columns=table_columns,
            show_limited_warning=True if total_rows > 100 else False
        )

    except Exception as e:
        print(f"Bulk prediction error: {e}")
        return render_template("bulk.html", error_msg=f"Error processing CSV: {str(e)}")

@app.route("/download", methods=["GET"])
def download():
    if os.path.exists(PREDICTIONS_OUTPUT):
        return send_file(
            PREDICTIONS_OUTPUT,
            mimetype="text/csv",
            as_attachment=True,
            download_name="churn_predictions.csv"
        )
    else:
        return redirect(url_for("bulk_predict"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

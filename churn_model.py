import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Paths
DATASET_DIR = "dataset"
DATASET_PATH = os.path.join(DATASET_DIR, "Telco-Customer-Churn.csv")
MODEL_PATH = "model.pkl"
COLUMNS_PATH = "model_columns.pkl"

# Check and download dataset if not exists
if not os.path.exists(DATASET_PATH):
    print("Dataset not found. Creating dataset directory and downloading Telco-Customer-Churn.csv...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    try:
        urllib.request.urlretrieve(url, DATASET_PATH)
        print("Dataset downloaded successfully.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise e

# 1. Load Telco-Customer-Churn.csv using pandas
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# 2. Drop the customerID column
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# 3. Convert TotalCharges to numeric using pd.to_numeric(..., errors='coerce') and fill NaN with 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

# 4. Encode the target column Churn: Yes -> 1, No -> 0
if "Churn" in df.columns:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
else:
    raise ValueError("Churn column not found in dataset")

# 5. Use pd.get_dummies() for all remaining categorical columns (drop_first=False so all columns are explicit)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Identify columns to get dummies for (categorical fields)
X = pd.get_dummies(X, drop_first=False)

# Convert all boolean columns created by get_dummies to integers (0/1) for compatibility
X = X.astype({col: int for col in X.select_dtypes(include=['bool']).columns})

# 6. Save the final column list (after get_dummies) to a file called model_columns.pkl using joblib
model_columns = list(X.columns)
joblib.dump(model_columns, COLUMNS_PATH)

# 8. Split into 80% train / 20% test using train_test_split(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train a RandomForestClassifier(n_estimators=100, random_state=42)
print("Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 10. Print: Accuracy, Confusion Matrix, and Classification Report
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\n--- MODEL PERFORMANCE METRICS ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# 11. Save model to model.pkl using joblib
joblib.dump(model, MODEL_PATH)

# 12. Print confirmation
print(f"\nModel saved as {MODEL_PATH}")
print(f"Columns saved as {COLUMNS_PATH}")

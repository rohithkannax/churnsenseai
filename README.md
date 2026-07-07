# ChurnSense AI — Customer Churn Prediction System

ChurnSense AI is a machine learning-based customer analytics platform designed to predict customer attrition (churn) for telecommunication providers. Built with Python, scikit-learn, and Flask, the application offers real-time churn likelihood metrics, explainable AI reasoning, and high-performance batch CSV file processing, all framed in a premium dark glassmorphism dashboard.

This project was built as a college final-year project to demonstrate practical machine learning modeling, explanation mechanics, and deployment interfaces.

---

## 🛠️ Prerequisites & Requirements

- **Python version**: Python 3.8 or higher is required.
- **Operating System**: Platform independent (supports Windows, macOS, and Linux).

---

## 🚀 Setup & Run Instructions

To run ChurnSense AI on your local machine, follow these steps:

### 1. Install Dependencies
Open your terminal in the project root folder and install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup & Training
The machine learning model requires the IBM Telco Customer Churn dataset to train.
*   **Automatic Download**: When you run the model training script, it will automatically download `Telco-Customer-Churn.csv` from a public IBM repository and save it to the `dataset/` directory.
*   **Manual Download (Alternative)**: You can manually download the dataset from [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it as `dataset/Telco-Customer-Churn.csv`.

Once the dataset is in place, train the classifier by running:
```bash
python churn_model.py
```
This script will:
- Load and clean the CSV data.
- Encode categorical features and build a column schema (`model_columns.pkl`).
- Split the dataset into 80% training and 20% validation.
- Train a `RandomForestClassifier` (100 estimators).
- Print accuracy scores and classification report tables.
- Save the trained binary classifier to `model.pkl`.

### 3. Start the Web Server
Launch the Flask backend server:
```bash
python app.py
```
The server will boot up locally at:
👉 **[http://localhost:5000](http://localhost:5000)** or **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📊 Dataset & Features Overview

The dataset contains customer demographic profiles, contract configurations, services subscribed, and billing metrics. Below are the 19 features evaluated:

1.  **gender**: Customer gender (Male, Female)
2.  **SeniorCitizen**: Whether the customer is a senior citizen (1, 0)
3.  **Partner**: Whether the customer has a partner (Yes, No)
4.  **Dependents**: Whether the customer has dependents (Yes, No)
5.  **tenure**: Number of months the customer has stayed with the company (0-72)
6.  **PhoneService**: Whether the customer has a phone service (Yes, No)
7.  **MultipleLines**: Whether the customer has multiple lines (Yes, No, No phone service)
8.  **InternetService**: Customer's internet service provider (DSL, Fiber optic, No)
9.  **OnlineSecurity**: Whether the customer has online security (Yes, No, No internet service)
10. **OnlineBackup**: Whether the customer has online backup (Yes, No, No internet service)
11. **DeviceProtection**: Whether the customer has device protection (Yes, No, No internet service)
12. **TechSupport**: Whether the customer has tech support (Yes, No, No internet service)
13. **StreamingTV**: Whether the customer has streaming TV (Yes, No, No internet service)
14. **StreamingMovies**: Whether the customer has streaming movies (Yes, No, No internet service)
15. **Contract**: The contract term of the customer (Month-to-month, One year, Two year)
16. **PaperlessBilling**: Whether the customer has paperless billing (Yes, No)
17. **PaymentMethod**: The customer's payment method (Electronic check, Mailed check, Bank transfer, Credit card)
18. **MonthlyCharges**: The amount charged to the customer monthly (numeric)
19. **TotalCharges**: The total amount charged to the customer (numeric)

---

## 💻 How to Use

### Single Prediction Page
1.  Navigate to the **Single Predict** tab in the top navigation bar.
2.  Select demographic parameters, services, contract types, and billing charges in the form inputs.
3.  Click **Analyze Churn Risk**.
4.  View results in the right-hand panel:
    *   **Result Badge**: Color-coded indicator displaying "Customer will CHURN" (Red) or "Customer will STAY" (Green).
    *   **Churn Probability**: Visual percentage gauge bar.
    *   **AI Risk Assessment**: A plain-English explanation generated dynamically explaining why the model reached that verdict based on customer risk profiles.
    *   **Top 5 Driving Model Features**: The model feature importances bar chart.

### Bulk Prediction Page
1.  Navigate to the **Bulk Predict** tab in the top navigation bar.
2.  Drag and drop or select a CSV file containing customer records formatted with the same features.
3.  Click **Process Batch and Predict**.
4.  The system evaluates all rows, computes statistics, and renders:
    *   **KPI Scorecards**: Total records processed, Churn count, Stay count, and cumulative Churn Rate.
    *   **Interactive Table**: Scrollable spreadsheet preview highlighting Churn (red rows) vs. Stay (green rows).
    *   **CSV Export**: Click **Download Predictions CSV** to save the processed dataset containing prediction outcomes and confidence columns.

---

## 📈 Model Performance Expectations

When trained on the IBM Telco Churn dataset, the RandomForest classifier achieves:
- **Model Accuracy**: ~79.5% - 80.5%
- **Metrics Evaluation**: Balanced Precision, Recall, and F1-Scores across stay vs. churn classes.

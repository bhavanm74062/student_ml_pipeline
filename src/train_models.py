import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, export_text

def train_all(data_path="data/student_data.csv"):
    df = pd.read_csv(data_path)
    features = ["study_hours", "attendance", "prev_grade", "assignments_done"]
    X = df[features]
    
    # Split indices consistently so all models evaluate on the exact same test students
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    
    # Feature Scaling (Crucial for Linear & Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # --- 1. LINEAR REGRESSION (Final Score) ---
    y_reg_train, y_reg_test = df.loc[train_idx, "final_score"], df.loc[test_idx, "final_score"]
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_reg_train)
    reg_preds = lin_reg.predict(X_test_scaled)
    
    print("\n--- 1. Linear Regression (Score Prediction) ---")
    print(f"R2 Score : {r2_score(y_reg_test, reg_preds):.4f}")
    print(f"RMSE     : {root_mean_squared_error(y_reg_test, reg_preds):.2f} points")
    joblib.dump(lin_reg, "models/linear_reg.pkl")

    # --- 2. LOGISTIC REGRESSION (Pass/Fail) ---
    y_bin_train, y_bin_test = df.loc[train_idx, "passed"], df.loc[test_idx, "passed"]
    log_reg = LogisticRegression()
    log_reg.fit(X_train_scaled, y_bin_train)
    bin_preds = log_reg.predict(X_test_scaled)
    
    print("\n--- 2. Logistic Regression (Pass/Fail) ---")
    print(f"Accuracy : {accuracy_score(y_bin_test, bin_preds)*100:.2f}%")
    joblib.dump(log_reg, "models/logistic_reg.pkl")

    # --- 3. DECISION TREE (Performance Tier) ---
    # Note: Decision Trees do not require scaled features
    y_cls_train, y_cls_test = df.loc[train_idx, "tier"], df.loc[test_idx, "tier"]
    dt_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_clf.fit(X_train, y_cls_train)
    cls_preds = dt_clf.predict(X_test)
    
    print("\n--- 3. Decision Tree (Tier Classification) ---")
    print(f"Accuracy : {accuracy_score(y_cls_test, cls_preds)*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_cls_test, cls_preds))
    print("Decision Rules Preview:\n", export_text(dt_clf, feature_names=features))
    joblib.dump(dt_clf, "models/decision_tree.pkl")

if __name__ == "__main__":
    train_all()
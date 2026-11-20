import joblib
import pandas as pd
from src.data_prep import generate_dataset
from src.train_models import train_all

def predict_new_student():
    scaler = joblib.load("models/scaler.pkl")
    lin_reg = joblib.load("models/linear_reg.pkl")
    log_reg = joblib.load("models/logistic_reg.pkl")
    dt_clf = joblib.load("models/decision_tree.pkl")

    print("\n=== Test a New Student Profile ===")
    hours = float(input("Weekly Study Hours (1-40): "))
    attendance = float(input("Attendance % (40-100): "))
    prev_grade = float(input("Previous Semester Grade (30-100): "))
    assignments = int(input("Assignments Completed (0-10): "))

    raw_data = pd.DataFrame([[hours, attendance, prev_grade, assignments]], 
                            columns=["study_hours", "attendance", "prev_grade", "assignments_done"])
    scaled_data = scaler.transform(raw_data)

    pred_score = lin_reg.predict(scaled_data)[0]
    pass_prob = log_reg.predict_proba(scaled_data)[0][1] * 100
    pred_tier = dt_clf.predict(raw_data)[0]

    print("\n--- Prediction Results ---")
    print(f"Estimated Final Score : {pred_score:.1f} / 100  (Linear Regression)")
    print(f"Probability of Passing: {pass_prob:.1f}%         (Logistic Regression)")
    print(f"Academic Advising Tier: {pred_tier}            (Decision Tree)")

if __name__ == "__main__":
    generate_dataset()
    train_all()
    predict_new_student()
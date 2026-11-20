import os
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

def generate_dataset(filepath="data/student_data.csv"):
    """
    Downloads and preprocesses the real UCI Student Performance dataset (student-por.csv),
    mapping its attributes to the unified 4-feature schema used across all models.
    """
    raw_csv_path = "data/student-por.csv"
    os.makedirs("data", exist_ok=True)

    # 1. Fetch official UCI dataset if not already cached locally
    if not os.path.exists(raw_csv_path):
        print("[INFO] Downloading real UCI Student Performance dataset...")
        req = urllib.request.urlopen(UCI_ZIP_URL, timeout=15)
        with zipfile.ZipFile(io.BytesIO(req.read())) as outer_zip:
            # The outer archive contains 'student.zip'
            with outer_zip.open("student.zip") as inner_zip_bytes:
                with zipfile.ZipFile(io.BytesIO(inner_zip_bytes.read())) as inner_zip:
                    with inner_zip.open("student-por.csv") as f_in, open(raw_csv_path, "wb") as f_out:
                        f_out.write(f_in.read())
        print(f"[INFO] Extracted raw dataset to {raw_csv_path}")

    # 2. Load raw semicolon-delimited UCI CSV
    raw_df = pd.read_csv(raw_csv_path, sep=";")

    # Filter out anomalous zero-dropout rows where G3 == 0 despite passing G2
    # (Common standard cleaning step for UCI Student Performance)
    raw_df = raw_df[raw_df["G3"] > 0].copy()

    # 3. Feature Engineering & Scale Alignment (0-20 scale -> 0-100 scale)
    # Map ordinal studytime (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h) to realistic weekly hours
    study_map = {1: 2.0, 2: 5.0, 3: 10.0, 4: 16.0}
    study_hours = raw_df["studytime"].map(study_map) + (raw_df["goout"].rsub(5) * 1.5)

    # Convert absences (out of ~90 semester days) into Attendance %
    attendance = (((90 - raw_df["absences"]) / 90.0) * 100).clip(40, 100)

    # Combine Period 1 (G1) and Period 2 (G2) grades, scaled from /20 to /100
    prev_grade = (((raw_df["G1"] + raw_df["G2"]) / 2.0) * 5.0).clip(30, 100)

    # Derive Academic Engagement / Assignments Completed (0-10 scale)
    engagement = (
        6
        + (raw_df["higher"] == "yes").astype(int) * 2
        + (raw_df["activities"] == "yes").astype(int) * 1
        + (raw_df["famsup"] == "yes").astype(int) * 1
        - raw_df["failures"].clip(0, 3)
    ).clip(0, 10)

    # Target 1: Final Grade (G3) scaled from /20 to /100
    final_score = (raw_df["G3"] * 5.0).clip(0, 100)

    # Target 2: Binary Pass/Fail (12/20 = 60/100)
    passed = (final_score >= 60).astype(int)

    # Target 3: Multi-class Advising Tier
    def get_tier(score):
        if score < 60:
            return "At-Risk"
        elif score < 80:
            return "Average"
        else:
            return "Honors"

    tier
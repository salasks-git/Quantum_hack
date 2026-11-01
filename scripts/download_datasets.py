import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_all():
    print("⏳ Downloading & standardizing clinical datasets to ./data/ ...")

    # 1. Wisconsin Breast Cancer (Oncology - 569 rows, 30 features)
    raw_bc = load_breast_cancer()
    df_bc = pd.DataFrame(raw_bc.data, columns=raw_bc.feature_names)
    df_bc["target"] = 1 - raw_bc.target  # 1 = Malignant (Disease), 0 = Benign (Healthy)
    df_bc.to_csv(os.path.join(DATA_DIR, "breast_cancer_wisconsin.csv"), index=False)
    print(f"✅ Saved data/breast_cancer_wisconsin.csv ({df_bc.shape[0]} rows, {df_bc.shape[1]-1} features)")

    # 2. Parkinson's Disease Vocal Biomarkers (Neurology - 195 rows, 22 features)
    try:
        park = fetch_openml(data_id=1488, as_frame=True, parser="auto")
    except Exception:
        park = fetch_openml(data_id=1488, as_frame=True, parser="liac-arff")
    df_park = park.data.select_dtypes(include=[np.number]).copy()
    df_park["target"] = (park.target.astype(str) == "2").astype(int).to_numpy()
    df_park.to_csv(os.path.join(DATA_DIR, "parkinsons_vocal.csv"), index=False)
    print(f"✅ Saved data/parkinsons_vocal.csv ({df_park.shape[0]} rows, {df_park.shape[1]-1} features)")

    # 3. Statlog Heart Disease (Cardiology - 270 rows, 13 features)
    heart = fetch_openml(data_id=53, as_frame=True, parser="auto")
    df_heart = heart.data.select_dtypes(include=[np.number]).copy()
    df_heart["target"] = (heart.target.astype(str) == "present").astype(int).to_numpy()
    df_heart.to_csv(os.path.join(DATA_DIR, "heart_disease_statlog.csv"), index=False)
    print(f"✅ Saved data/heart_disease_statlog.csv ({df_heart.shape[0]} rows, {df_heart.shape[1]-1} features)")

    # 4. Pima Indians Diabetes (Endocrinology - 768 rows, 8 features)
    diab = fetch_openml(data_id=37, as_frame=True, parser="auto")
    df_diab = diab.data.select_dtypes(include=[np.number]).copy()
    df_diab["target"] = (diab.target.astype(str) == "tested_positive").astype(int).to_numpy()
    df_diab.to_csv(os.path.join(DATA_DIR, "diabetes_pima.csv"), index=False)
    print(f"✅ Saved data/diabetes_pima.csv ({df_diab.shape[0]} rows, {df_diab.shape[1]-1} features)")

    # 5. Vertebral Column 2C (Orthopedics - 310 rows, 6 biomechanical features)
    # OpenML 1524 = UCI "column_2C": class '1' = Abnormal (disc hernia / spondylolisthesis), '2' = Normal
    try:
        vert = fetch_openml(data_id=1524, as_frame=True, parser="auto")
    except Exception:
        vert = fetch_openml(data_id=1524, as_frame=True, parser="liac-arff")
    df_vert = vert.data.astype(float).copy()
    df_vert.columns = [
        "pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle",
        "sacral_slope", "pelvic_radius", "degree_spondylolisthesis",
    ]
    df_vert["target"] = (vert.target.astype(str) == "1").astype(int).to_numpy()
    df_vert.to_csv(os.path.join(DATA_DIR, "vertebral_column.csv"), index=False)
    print(f"✅ Saved data/vertebral_column.csv ({df_vert.shape[0]} rows, {df_vert.shape[1]-1} features)")

    print("🎉 All 5 datasets downloaded, cleaned, and ready for offline QML training!")

if __name__ == "__main__":
    download_all()

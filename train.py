import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ==== 1. Đọc dữ liệu ====
df = pd.read_csv("Heart.csv")

# Xử lý dữ liệu thiếu
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Xóa cột không cần thiết (nếu có)
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)


# Encode cột phân loại
label_cols = ["ChestPain", "RestECG", "Slope", "Thal", "AHD"]
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Tách X, y
X = df.drop("AHD", axis=1)
y = df["AHD"]

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ==== 2. Logistic Regression ====
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# ==== 3. Random Forest ====
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ==== 4. Đánh giá ====
models = {
    "Logistic Regression": y_pred_log,
    "Random Forest": y_pred_rf
}

results = []
for name, y_pred in models.items():
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append([name, acc, prec, rec, f1])

df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print(df_results)


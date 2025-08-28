import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pathlib

st.set_page_config(page_title="Dự đoán Bệnh Tim", page_icon="💓", layout="wide")

def local_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("static/style.css")

st.markdown("<div class='dashboard-header'><h1>💓 Ứng dụng dự đoán bệnh tim</h1><p>Sử dụng Logistic Regression & Random Forest</p></div>", unsafe_allow_html=True)

# ==================== 1) Load & Preprocess ====================
@st.cache_data
def load_and_prepare(csv_path: str = "Heart.csv"):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.replace(["?", "NA", "na", "None", "null", ""], np.nan, inplace=True)

    cat_cols = [c for c in ["ChestPain", "RestECG", "Slope", "Thal", "AHD"] if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if num_cols:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    encoders = {}
    for c in ["ChestPain", "RestECG", "Slope", "Thal", "AHD"]:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le

    if "AHD" not in df.columns:
        raise ValueError("Không tìm thấy cột nhãn 'AHD' trong Heart.csv")

    X = df.drop(columns=["AHD"])
    y = df["AHD"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    stats = {c: {"min": float(np.nanmin(df[c])),
                 "max": float(np.nanmax(df[c])),
                 "median": float(np.nanmedian(df[c]))} for c in X.columns}

    return df, X, y, X_scaled, scaler, encoders, stats

# Load data
df, X, y, X_scaled, scaler, encoders, stats = load_and_prepare("Heart.csv")
feature_names = list(X.columns)

# ==================== 2) Train models ====================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)

# ==================== 3) DASHBOARD BODY ====================
tab_pred, tab_compare = st.tabs(["🔍 Dự đoán cá nhân", "📊 So sánh mô hình"])

with tab_pred:
    st.sidebar.header("📝 Nhập thông tin bệnh nhân")

    def med(col, as_int=False):
        m = stats[col]["median"]
        return int(round(m)) if as_int else float(m)

    inputs = {}
    inputs["Age"] = st.sidebar.slider("Tuổi", int(stats["Age"]["min"]), int(stats["Age"]["max"]), med("Age", True))
    inputs["Sex"] = st.sidebar.radio("Giới tính", [0, 1], index=int(med("Sex", True)), format_func=lambda x: "Nam" if x==1 else "Nữ")

    cp_classes = encoders["ChestPain"].classes_.tolist()
    cp_label = st.sidebar.selectbox("Đau ngực (asymptomatic=không triệu chứng, typical angina=điển hình, nontypical angina=không điển hình, nonanginal=không phải đau thắt ngực)", cp_classes)
    inputs["ChestPain"] = int(encoders["ChestPain"].transform([cp_label])[0])

    inputs["RestBP"] = st.sidebar.slider("Huyết áp nghỉ (mmHg)", int(stats["RestBP"]["min"]), int(stats["RestBP"]["max"]), med("RestBP", True))
    inputs["Chol"] = st.sidebar.slider("Cholesterol (mg/dl)", int(stats["Chol"]["min"]), int(stats["Chol"]["max"]), med("Chol", True))
    inputs["Fbs"] = st.sidebar.radio("Đường huyết lúc đói >120mg/dl (0=Không, 1=Có)", [0, 1], index=int(med("Fbs", True)))

    recg_classes = encoders["RestECG"].classes_.tolist()
    recg_label = st.sidebar.selectbox("Điện tâm đồ khi nghỉ (0=Bình thường, 1=Bất thường ST-T, 2=Phì đại thất trái)", recg_classes)
    inputs["RestECG"] = int(encoders["RestECG"].transform([recg_label])[0])

    inputs["MaxHR"] = st.sidebar.slider("Nhịp tim tối đa", int(stats["MaxHR"]["min"]), int(stats["MaxHR"]["max"]), med("MaxHR", True))
    inputs["ExAng"] = st.sidebar.radio("Đau thắt ngực khi gắng sức (0=Không, 1=Có)", [0, 1], index=int(med("ExAng", True)))
    inputs["Oldpeak"] = st.sidebar.number_input("Độ chênh ST", value=med("Oldpeak"), step=0.1, format="%.1f")

    slope_classes = encoders["Slope"].classes_.tolist()
    slope_label = st.sidebar.selectbox("Độ dốc đoạn ST (1 = lên dốc, 2 = phẳng, 3 = xuống dốc)", slope_classes)
    inputs["Slope"] = int(encoders["Slope"].transform([slope_label])[0])

    ca_min, ca_max = int(stats["Ca"]["min"]), int(stats["Ca"]["max"])
    inputs["Ca"] = st.sidebar.selectbox("Số mạch vành chính", list(range(ca_min, ca_max + 1)), index=0)

    thal_classes = encoders["Thal"].classes_.tolist()
    thal_label = st.sidebar.selectbox("Thalassemia (normal = bình thường, fixed defect = khiếm khuyết cố định, reversible defect = khiếm khuyết hồi phục)", thal_classes)
    inputs["Thal"] = int(encoders["Thal"].transform([thal_label])[0])

    model_choice = st.radio("🔎 Chọn mô hình", ["Logistic Regression", "Random Forest"])

    if st.button("🚀 Dự đoán"):
        row = np.array([inputs[col] for col in feature_names]).reshape(1, -1)
        row_scaled = scaler.transform(row)

        if model_choice == "Logistic Regression":
            proba = float(log_reg.predict_proba(row_scaled)[0][1])
            pred = int(proba >= 0.5)
        else:
            proba = float(rf.predict_proba(row_scaled)[0][1])
            pred = int(proba >= 0.5)

        st.subheader("🩺 Kết quả")
        st.dataframe(pd.DataFrame([inputs], columns=feature_names))

        if pred == 1:
            st.markdown(f"<div class='result-box' style='background:#ffe2e2;color:#b00020;'>⚠️ Nguy cơ bệnh tim — Xác suất {proba:.2f}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box' style='background:#e6ffed;color:#006400;'>✅ Không có nguy cơ bệnh tim — Xác suất {proba:.2f}</div>", unsafe_allow_html=True)

with tab_compare:
    y_pred_log = log_reg.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    df_scores = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": accuracy_score(y_test, y_pred_log),
         "Precision": precision_score(y_test, y_pred_log),
         "Recall": recall_score(y_test, y_pred_log),
         "F1-Score": f1_score(y_test, y_pred_log)},
        {"Model": "Random Forest", "Accuracy": accuracy_score(y_test, y_pred_rf),
         "Precision": precision_score(y_test, y_pred_rf),
         "Recall": recall_score(y_test, y_pred_rf),
         "F1-Score": f1_score(y_test, y_pred_rf)}
    ])

    st.subheader("📊 So sánh mô hình")
    st.dataframe(df_scores.set_index("Model"))
    st.bar_chart(df_scores.set_index("Model"))

# ==================== FOOTER ====================
st.markdown("<div class='dashboard-footer'>© 2025 Ứng dụng dự đoán bệnh tim | Thực hiện bởi nhóm Qui Chen</div>", unsafe_allow_html=True)

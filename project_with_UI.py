import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Default Prediction", layout="centered")
st.title("Predict Default Risk Based on User Input")

# Load and train models
@st.cache_data
def load_data_and_models():
    df = pd.read_csv("original.csv")

    # Clean data
    if 'clientid' in df.columns:
        df = df.drop(columns=['clientid'])
    df = df[df['age'] >= 0]

    X = df.drop(columns=['default'])
    y = df['default']
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier()
    svc = LinearSVC(max_iter=1000)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    svc.fit(X_train, y_train)

    acc_lr = accuracy_score(y_test, lr.predict(X_test))
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    acc_svc = accuracy_score(y_test, svc.predict(X_test))

    return lr, rf, svc, acc_lr, acc_rf, acc_svc, feature_cols

lr_model, rf_model, svc_model, acc_lr, acc_rf, acc_svc, feature_cols = load_data_and_models()

# Show input form only for the required features
st.subheader("Enter Customer Info")

# Default values can be adjusted as per your data range
user_inputs = {}
for col in feature_cols:
    if col == 'income':
        user_inputs[col] = st.slider(f"{col.capitalize()}", 0, 100000, 30000)
    elif col == 'loan':
        user_inputs[col] = st.slider(f"{col.capitalize()}", 0, 50000, 10000)
    elif col == 'age':
        user_inputs[col] = st.slider(f"{col.capitalize()}", 18, 100, 30)
    else:
        # For any extra columns like 'employ', 'married', etc.
        user_inputs[col] = st.number_input(f"{col.capitalize()}", value=0)

# Convert to DataFrame and match training column order
input_df = pd.DataFrame([user_inputs])[feature_cols]

if st.button("Predict Default Risk"):
    pred_lr = lr_model.predict(input_df)[0]
    pred_rf = rf_model.predict(input_df)[0]
    pred_svc = svc_model.predict(input_df)[0]

    def show_result(name, pred, acc):
        status = "Default" if pred == 1 else "No Default"
        st.markdown(f"**{name}** — Prediction: {status} | Accuracy: **{acc:.2%}**")

    st.subheader("Prediction Results")
    show_result("Logistic Regression", pred_lr, acc_lr)
    show_result("Random Forest", pred_rf, acc_rf)
    show_result("Linear SVC", pred_svc, acc_svc)

import streamlit as st
import numpy as np
import pickle


# Background image using custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.pinimg.com/736x/d4/05/55/d405555eb81ce6c883ca7eee76b487a5.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load all models and scalers
def load_model_and_scaler(model_path, scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Sidebar - Disease selection
st.sidebar.header("Choose Disease")
selected_disease = st.sidebar.selectbox(
    "Select Disease", 
    ["Parkinson's Disease", "Heart Disease", "Breast Cancer", "Diabetes"]
)

# Load the correct model and scaler
if selected_disease == "Parkinson's Disease":
    model, scaler = load_model_and_scaler('pkl/best_model_P.pkl', 'Scaler/scaler_P.pkl')
elif selected_disease == "Heart Disease":
    model, scaler = load_model_and_scaler('pkl/best_model.pkl', 'Scaler/scaler.pkl')
elif selected_disease == "Breast Cancer":
    model, scaler = load_model_and_scaler('pkl/best_model_can.pkl', 'Scaler/scaler_can.pkl')
else:
    model, scaler = load_model_and_scaler('pkl/best_model_Dia.pkl', 'Scaler/scaler_Dia.pkl')

# Feature inputs for each disease
features = {
    "Parkinson's Disease": ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'spread1', 'spread2', 'DFA', 'PPE'],
    "Heart Disease": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
    "Breast Cancer": ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'],
    "Diabetes": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
}

# Title and Header
st.title("🧠 AI-Powered Disease Prediction System")
st.header(f"{selected_disease} Prediction")

# Input form
input_data = []
for feature in features[selected_disease]:
    value = st.number_input(f"{feature}", step=0.1, format="%.2f")
    input_data.append(value)

# Prediction
if st.button("Predict"):
    scaled_input = scaler.transform([input_data])
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error(f"⚠️ The prediction indicates **presence** of {selected_disease}.")
    else:
        st.success(f"✅ The prediction indicates **no sign** of {selected_disease}.")

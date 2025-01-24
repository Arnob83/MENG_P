import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from groq import Groq
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import shap
from shap.maskers import Independent
import requests

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Smart Loan Application",page_icon='🏦')
# Hardcoded credentials
USERNAME = "1"
PASSWORD = "1"

# Function for login screen
def login_screen():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    # Embed custom CSS for full-screen video background
    st.markdown(
        """
        <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        video {
            position: fixed;
            top: 0;
            left: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;
            object-fit: cover;
        }
        .stApp {
            background: transparent;
        }
        </style>
        <video autoplay muted loop>
            <source src="https://cdn.pixabay.com/video/2021/10/30/93956-641767616_large.mp4" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )
    st.title("🏦Smart Loan Application")
    st.subheader("Your one-stop solution for all financial burden")
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            
            st.session_state["logged_in"] = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password.")


# Main loan prediction and recommendation dashboard
def main_app():
    
   # Load Groq API key securely from Streamlit secrets
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("API key not found! Please set it in Streamlit Secrets.")
    st.stop()

# Set Groq API client
client = Groq(
    api_key=api_key,
)

    
  # Import requests to fetch files from URLs

# Define paths
model_path = "https://raw.githubusercontent.com/Arnob83/MENG_P/main/Logistic_Regression_model.pkl"
scaler_path = "https://raw.githubusercontent.com/Arnob83/MENG_P/main/scaler.pkl"
data_path = "https://raw.githubusercontent.com/Arnob83/MENG_P/main/X_train_scaled.pkl"

# Load models and data from URLs
@st.cache_resource
def load_model_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request fails
    return pickle.loads(response.content)

@st.cache_resource
def load_scaler_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return pickle.loads(response.content)

@st.cache_resource
def load_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return pickle.loads(response.content)

# Load resources
scaler = load_scaler_from_url(scaler_path)
model = load_model_from_url(model_path)
data = load_data_from_url(data_path)


    # Define feature names in the correct order (as used during model training)
    feature_names = [
        'Credit_History', 'Education', 'ApplicantIncome', 
        'CoapplicantIncome', 'Loan_Amount_Term', 
        'Property_Area', 'Gender'
    ]

    # User input
    st.header("Provide Details for Loan Prediction")

    user_input = {}

    # Customer details
    user_input['Customer_Name'] = st.text_input("Customer Name:")

    # Dropdown for credit history
    credit_history = st.selectbox("Do you have a good credit history?", ['Yes', 'No'])
    user_input['Credit_History'] = 1 if credit_history == 'Yes' else 0

    # Dropdown for education
    education = st.selectbox("Education Level:", ['Graduate', 'Non-Graduate'])
    user_input['Education'] = 0 if education == 'Graduate' else 1

    # Slider for applicant and co-applicant income
    user_input['ApplicantIncome'] = st.number_input("Applicant Income ($/Monthly):", value=None, placeholder="Enter Income in Dollar")
    user_input['CoapplicantIncome'] = st.number_input("Co-applicant Income ($/Monthly):", value=None, placeholder="Enter Income in Dollar")

    # Dropdown for gender
    gender = st.selectbox("Gender:", ['Male', 'Female'])
    user_input['Gender'] = 1 if gender == 'Male' else 0

    # Loan amount term
    user_input['Loan_Amount_Term'] = st.number_input("Enter Loan Amount Term (in months):", value=None, step=1, placeholder="Enter Months in number...", label_visibility="visible", help="Enter Loan term amount in months. The value should be a positive integer. Example - 6, 12, 18, 24, 36 etc.")

    # Dropdown for property area
    property_area = st.selectbox("Property Area:", ['Urban', 'Semiurban', 'Rural'])
    property_area_mapping = {'Urban': 0.6584158416, 'Semiurban': 0.7682403433, 'Rural': 0.6145251397}
    user_input['Property_Area'] = property_area_mapping[property_area]

    # Extra fields
    user_input['Married'] = st.selectbox("Marital Status:", ['Married', 'Unmarried'])

    
    user_input['Dependents'] = st.number_input("Number of Dependents:", min_value=0, step=1)

    user_input['Employment'] = st.selectbox("Self-Employment :", ['Yes', 'No'])

    # Loan amount slider
    user_input['Loan_Amount'] = st.number_input("Loan Amount (1000 x $):", value=None, placeholder="Enter Loan amount as per 1000$...")

    # Convert user input into a dataframe with the correct feature order
    feature_values = {
        'Credit_History': user_input['Credit_History'],
        'Education': user_input['Education'],
        'ApplicantIncome': user_input['ApplicantIncome'],
        'CoapplicantIncome': user_input['CoapplicantIncome'],
        'Loan_Amount_Term': user_input['Loan_Amount_Term'],
        'Property_Area': user_input['Property_Area'],
        'Gender': user_input['Gender'],
    }
    user_input_df = pd.DataFrame([feature_values])[feature_names]

    # Prediction button
    if st.button("Predict"):
        # Preprocess input
        scaled_features = scaler.transform(user_input_df[['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term']])
        user_input_df[['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term']] = scaled_features

        # Predict
        prediction = model.predict(user_input_df)
        prediction_proba = model.predict_proba(user_input_df)[:, 1]

        # Display prediction results
        st.subheader("Prediction Result")
        st.write(f"**Loan Status:** {'Approved' if prediction[0] == 1 else 'Declined'}")
        st.write(f"**Approval Probability:** {prediction_proba[0]:.2f}")

        
        st.subheader("Feature Importance")
        masker = shap.maskers.Independent(Data)
        explainer = shap.LinearExplainer(model, masker)
        shap_values = explainer(user_input_df)
        shap.summary_plot(shap_values, user_input_df, plot_type="bar", show=False)
        st.pyplot(plt.gcf())
        plt.clf()

       

    





        # Suggestion and recommendation
        st.subheader("Suggestions and Recommendations")

        # Create prompt for Groq
        prompt = (
            f"The user named {user_input['Customer_Name']} - provided the following details for a loan application:\n\n"
            f"Credit History: {user_input['Credit_History']} - 0 stands for bad credit history and 1 stands for good credit history,\n"
            f"Education: {user_input['Education']} - 1 stands for non-graduate and 0 stands for graduate,\n"
            f"Applicant Income: {user_input['ApplicantIncome']} - Applicant Income is provided in Dollar per month ,\n"
            f"Coapplicant Income: {user_input['CoapplicantIncome']} - CoApplicant Income is provided in Dollar per month,\n"
            f"Loan Amount Term: {user_input['Loan_Amount_Term']} - Loan Tenture is Entered as total months,\n"
            f"Property Area: {user_input['Property_Area']}, -  Property type is defined as a number as followed - '0.6584158416'means Urban, '0.7682403433' means Semiurban , '0.6145251397' means Rural  \n"
            f"Gender: {user_input['Gender']}\n\n"
            f"The loan prediction model determined the loan status as "
            f"{'Approved' if prediction[0] == 1 else 'Declined'} with a probability of {prediction_proba[0]:.2f}.\n\n"
            f"Please provide reasons for this decision and suggest ways to improve approval chances if declined."
            f"You dont need to add user exact information. Please state that using english word in bullet points"
            f"Based on this, please explain the main reason factor for this decision in simple terms according to user input, and provide suggestions on how the user can modify their inputs to improve the chances of loan approval in future if loan is rejected. No suggestion is needed if the loan is approved.\n"
            f"Give the response in structured format with two sections - Reasons and Suggestions."
        )

        try:
            # Groq API response
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
            )

            st.write(chat_completion.choices[0].message.content.strip())
            st.button("Logout", type='primary', on_click=lambda: st.session_state.update({"logged_in": False}))

        except Exception as e:
            st.error(f"An error occurred while fetching recommendations: {e}")

# Check if user is logged in
#if "logged_in" not in st.session_state:
    #st.session_state["logged_in"] = False

#if not st.session_state["logged_in"]:
    #login_screen()
#else:
    #main_app()

    # Check if user is logged in
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_screen()
else:
    # Add logout button at the top of the main app
    main_app()

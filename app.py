import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from groq import Groq
import os
import shap
import matplotlib.pyplot as plt
import sqlite3
import io
import requests





def save_to_sqlite(data):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect("user_history.db")
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_history (
            Customer_Name TEXT,
            Credit_History INTEGER,
            Education INTEGER,
            ApplicantIncome REAL,
            CoapplicantIncome REAL,
            Loan_Amount_Term INTEGER,
            Property_Area REAL,
            Gender INTEGER,
            Loan_Amount REAL,
            Marital_Status TEXT,
            Dependents INTEGER,
            Self_Employment TEXT,
            Loan_Status TEXT,
            Approval_Probability REAL
        )
    """)

    # Insert data into the table
    data.to_sql("user_history", conn, if_exists="append", index=False)

    # Commit and close connection
    conn.commit()
    conn.close()

def clear_database(database_path):
    """
    Deletes all data from the SQLite database table 'user_history'.
    """
    try:
        with sqlite3.connect(database_path) as conn:
            conn.execute("DELETE FROM user_history")  # Deletes all rows in the table
            conn.commit()  # Commit changes to save the deletions
            print("Database cleared successfully.")
    except sqlite3.Error as e:
        print(f"Error clearing the database: {e}")



st.set_page_config(page_title="Smart Loan Application",page_icon='🏦')


CREDENTIALS = {
    "user": {"username": "user@gmail.com", "password": "12345"},
    "admin": {"username": "admin@gmail.com", "password": "12345"},
}

# Initialize session states
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_role" not in st.session_state:
    st.session_state["user_role"] = None
if "user_history" not in st.session_state:
    # Initialize an empty DataFrame to store user requests history
    st.session_state["user_history"] = pd.DataFrame(columns=[
        "Customer_Name", "Credit_History", "Education", "ApplicantIncome", 
        "CoapplicantIncome", "Loan_Amount_Term", "Property_Area", "Gender", 
        "Loan_Amount", "Marital_Status", "Dependents", "Self_Employment", 
        "Loan_Status", "Approval_Probability"
    ])

# Function for login screen
def login_screen():
    
    
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
       
        """,
        unsafe_allow_html=True
    )
    st.title("🏦Smart Loan Application")
    st.subheader("Your one-stop solution for all financial burden")
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Check user credentials
        if username == CREDENTIALS["user"]["username"] and password == CREDENTIALS["user"]["password"]:
            st.session_state["logged_in"] = True
            st.session_state["user_role"] = "user"
            st.success("User login successful!")
        elif username == CREDENTIALS["admin"]["username"] and password == CREDENTIALS["admin"]["password"]:
            st.session_state["logged_in"] = True
            st.session_state["user_role"] = "admin"
            st.success("Admin login successful!")
        else:
            st.error("Invalid username or password.")



# Main loan prediction and recommendation dashboard

    
def user_app():
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
    # The rest of the code for the main_app() function goes here


        
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
    user_input['Marital_Status'] = st.selectbox("Marital_Status:", ['Married', 'Unmarried'])

    
    user_input['Dependents'] = st.number_input("Number of Dependents:", min_value=0, step=1)

    user_input['Self_Employment'] = st.selectbox("Self_Employment :", ['Yes', 'No'])

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
        loan_status = "Approved" if prediction[0] == 1 else "Declined"

        # Display prediction results
        st.subheader("Prediction Result")
        st.write(f"**Loan Status:** {loan_status}")
        st.write(f"**Approval Probability:** {prediction_proba[0]:.2f}")

       
        masker = shap.maskers.Independent(data)
        explainer = shap.LinearExplainer(model, masker)
        shap_values = explainer(user_input_df)
        shap.summary_plot(shap_values, user_input_df, plot_type="bar", show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        
        

        # Save user request and prediction to history
        user_input['Loan_Status'] = loan_status
        user_input['Approval_Probability'] = prediction_proba[0]
        selected_columns = [
        "Customer_Name", "Credit_History", "Education", "ApplicantIncome",
        "CoapplicantIncome", "Loan_Amount_Term", "Property_Area", "Gender",
        "Loan_Amount", "Marital_Status", "Dependents", "Self_Employment",
        "Loan_Status", "Approval_Probability"
    ]
    # Create a cleaned DataFrame with the selected columns
        cleaned_user_input = {key: user_input[key] for key in selected_columns if key in user_input}
        cleaned_user_df = pd.DataFrame([cleaned_user_input])

    # Update the session state
        st.session_state["user_history"] = pd.concat(
        [st.session_state["user_history"], cleaned_user_df],
        ignore_index=True
    )

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
            f"{loan_status} with a probability of {prediction_proba[0]:.2f}.\n\n"
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
            
        except Exception as e:
            st.error(f"An error occurred while fetching recommendations: {e}")
        col1, col2 = st.columns([1, 1]) 
        with col1:
            st.button("Logout", type='primary', on_click=lambda: st.session_state.update({"logged_in": False}))

        with col2:        
            if st.button("New Request", type='secondary'):
                # Clear user input session state
                st.session_state["user_history"] = pd.DataFrame(columns=[
                    "Customer_Name", "Credit_History", "Education", "ApplicantIncome",
                    "CoapplicantIncome", "Loan_Amount_Term", "Property_Area", "Gender",
                    "Loan_Amount", "Marital_Status", "Dependents", "Self_Employment",
                    "Loan_Status", "Approval_Probability"
                        ])
                
                # Reset user inputs (the session state for user input data)
                st.session_state["user_input"] = {}

                # Reset any other variables if needed
                st.session_state["logged_in"] = False  # Optional: Reset login status (if you want to reset user session)
                
                # Trigger rerun to clear the form fields and reset the app state
                st.experimental_rerun()

                # Display a success message
                st.success("All inputs have been cleared. You can now make a new loan request.")
def admin_app():
    st.header("Admin Dashboard")
    st.write("View and manage user loan application history.")

    # Display user history
    if not st.session_state["user_history"].empty:
        # Apply the required mappings to the columns
        user_history = st.session_state["user_history"]

        # Map the values to human-readable strings
        user_history["Credit_History"] = user_history["Credit_History"].map({1: "Yes", 0: "No"})
        user_history["Education"] = user_history["Education"].map({0: "Graduate", 1: "Undergraduate"})
        user_history["Property_Area"] = user_history["Property_Area"].map({
            0.6584158416: "Urban", 0.7682403433: "Semiurban", 0.6145251397: "Rural"
        })
        user_history["Gender"] = user_history["Gender"].map({1: "Male", 0: "Female"})

        # Specify columns to display
        columns_to_display = [
            "Customer_Name", "Credit_History", "Education", "ApplicantIncome",
            "CoapplicantIncome", "Loan_Amount_Term", "Property_Area", "Gender",
            "Loan_Amount", "Marital_Status", "Dependents", "Self_Employment",
            "Loan_Status", "Approval_Probability"
        ]
        
        
        #save_to_sqlite(user_history)
        save_to_sqlite(st.session_state["user_history"])


        # Export the SQLite database file for download
        sqlite_file = io.BytesIO()
        with open("user_history.db", "rb") as db_file:
            sqlite_file.write(db_file.read())
        sqlite_file.seek(0)

        # Provide download button for SQLite database
        st.download_button(
            label="Download Database (SQLite)",
            data=sqlite_file,
            file_name="user_history.db",
            mime="application/x-sqlite3"
        )

        # Display the dataframe with only the relevant columns
        st.dataframe(user_history[columns_to_display])  # Filter only the columns to display
        
        

    else:
        st.write("No user requests history available.")

        # Clear both Streamlit cache and SQLite data

    if st.button("Clear Database"):
        clear_database("user_history.db")  # Function to clear the database
        st.session_state["user_history"] = pd.DataFrame(columns=[
            "Customer_Name", "Credit_History", "Education", "ApplicantIncome",
            "CoapplicantIncome", "Loan_Amount_Term", "Property_Area", "Gender",
            "Loan_Amount", "Marital_Status", "Dependents", "Self_Employment",
            "Loan_Status", "Approval_Probability"
        ])  # Clear in-memory history as well
        st.success("Database and session history cleared.")
        


    st.button("Logout", type='primary', on_click=lambda: st.session_state.update({"logged_in": False}))
    
    

# Check if user is logged in
if not st.session_state["logged_in"]:
    login_screen()
else:
    if st.session_state["user_role"] == "user":
        user_app()
    elif st.session_state["user_role"] == "admin":
        admin_app()


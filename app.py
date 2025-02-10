import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("NEW_Final.pkl")

# Streamlit App Title
st.title("🚩 Fake Social Media Account Detector")
st.write("Upload a CSV file to detect fake accounts.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Ensure required columns exist
    required_columns = ["username", "report_count", "follower_following_ratio", "verification_status", "engagement_pattern"]
    if not all(col in df.columns for col in required_columns):
        st.error("CSV file must contain the required columns: username, report_count, follower_following_ratio, verification_status, engagement_pattern")
    else:
        # Convert verification status to numeric
        df["verification_status"] = df["verification_status"].map({"Verified": 0, "Not Verified": 1})
        
        # Drop unnecessary columns before prediction
        df = df.drop(columns=["label"], errors="ignore")  # Drop 'label' if present

        # Predict using the trained model
        predictions = model.predict(df.drop(columns=["username"]))

        # Get fake accounts
        fake_accounts = df.loc[predictions == 1, "username"].tolist()

        # Display the results
        if fake_accounts:
            st.error("🚩 Fake Accounts Detected:")
            st.write(fake_accounts)
        else:
            st.success("✅ No fake accounts detected!")

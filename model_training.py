import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/FINAL.csv')

# Encode categorical feature (Verification Status)
label_encoder = LabelEncoder()
df['verification_status'] = label_encoder.fit_transform(df['verification_status'])

# Define thresholds for detection
thresholds = {
    "report_count": 5,
    "follower_following_ratio": 3.0,
    "verification_status": 1,  # 1 for Not Verified, 0 for Verified
    "engagement_pattern": 0.3,
}

# Extract features and target variable
X = df[['report_count', 'follower_following_ratio', 'verification_status', 'engagement_pattern']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred))

# Saving the trained model
joblib.dump(model, 'Final.pkl')
print("Model saved as Final.pkl")

# Creating a manual testing dataset
manual_test_data = pd.DataFrame({
    "username": ["user_201", "user_202", "user_203", "user_204", "user_205"],
    "report_count": [2, 8, 1, 7, 3],
    "follower_following_ratio": [1.5, 4.5, 2.0, 3.9, 0.8],
    "verification_status": ["Verified", "Not Verified", "Verified", "Not Verified", "Verified"],
    "engagement_pattern": [0.7, 0.2, 0.8, 0.1, 0.6]
})

# Encode verification status
manual_test_data['verification_status'] = label_encoder.transform(manual_test_data['verification_status'])

# Make predictions on manual testing data
manual_test_pred = model.predict(manual_test_data.drop(columns=["username"]))

# Identify fake accounts
fake_accounts = manual_test_data[manual_test_pred == 1]['username'].tolist()
print("Fake Accounts Detected:", fake_accounts)

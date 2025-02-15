##  Project Overview
This project detects "fake social media accounts" using Machine Learning (Random Forest classifier) and a "Streamlit-based UI". It classifies accounts as "fake or real" based on features like:
- Report count
- Follower/Following ratio
- Verification status
- Engagement patterns
- Comment text analysis

The app supports  "CSV uploads" for batch detection.

## 🛠️ Technologies Used
- Python (Scikit-Learn, TensorFlow, Pandas, NumPy)
- Machine Learning (Random Forest)
- NLP (Comment text analysis)
- Streamlit (User Interface)

##  Features
✅ Detects fake accounts based on predefined thresholds & ML models
✅ Displays "fake account usernames" in the UI

##  Installation & Setup
### 🔹 Prerequisites
Ensure you have **Python 3.8+**, and install dependencies:
bash
pip install -r requirements.txt


### 🔹 Run the Streamlit App
bash
streamlit run app.py


## 🖥️ Usage Guide
 Upload a CSV file → View fake account usernames

## 📂 Project Structure
├── app.py              # Streamlit UI
├── model_training.py   # ML Model Training Script
├── Final.csv         # Training Data
├── testing_dataset.csv # Testing Data
├── requirements.txt    # Required Python Packages


##  Testing
Upload "testing_dataset.csv" to verify that the model correctly flags fake accounts.

## 🌟 Future Improvements
- Improve dataset quality by collecting real-world data.
- Optimize the ML model to achieve "higher than 60% accuracy".
- Experiment with **advanced models like transformers (BERT/GPT)** for better NLP analysis.
- Develop a "mobile-friendly version" of the UI

## 📜 License
This project is open-source under the "MIT License".

## 📬 Contact
For questions or contributions, reach out via **GitHub Issues**.

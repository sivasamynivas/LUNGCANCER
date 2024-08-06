import pandas as pd
import numpy as np
import streamlit as st
from collections import Counter
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("dataseter.csv")
 
df.dropna(inplace=True)
df['GENDER'] = df['GENDER'].map({"M":1,"F":0})
df['SMOKING'] = df['SMOKING'].map({"Yes":1,"No":0})
df['ANXIETY'] = df['ANXIETY'].map({"Yes":1,"No":0})
df['PEER_PRESSURE'] = df['PEER_PRESSURE'].map({"Yes":1,"No":0})
df['CHRONIC_DISEASE'] = df['CHRONIC_DISEASE'].map({"Yes":1,"No":0})
df['FATIGUE'] = df['FATIGUE'].map({"Yes":1,"No":0})
df['ALLERGY'] = df['ALLERGY'].map({"Yes":1,"No":0})
df['WHEEZING'] = df['WHEEZING'].map({"Yes":1,"No":0})
df['ALCOHOL_CONSUMING'] = df['ALCOHOL_CONSUMING'].map({"Yes":1,"No":0})
df['COUGHING'] = df['COUGHING'].map({"Yes":1,"No":0})
df['SHORTNESS_OF_BREATH'] = df['SHORTNESS_OF_BREATH'].map({"Yes":1,"No":0})
df['SWALLOWING_DIFFICULTY'] = df['SWALLOWING_DIFFICULTY'].map({"Yes":1,"No":0})
df['CHEST_PAIN'] = df['CHEST_PAIN'].map({"Yes":1,"No":0})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({"YES":1,"NO":0})
df['YELLOW_FINGERS'] = df['YELLOW_FINGERS'].map({"Yes":1,"No":0})

# Create a StandardScaler object
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

skewness_values = df.skew()
# Separate features and target variable
# building for classification model
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# Instantiate SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Fit and apply SMOTE
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print class distribution after oversampling
# Print class distribution before oversampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a Random Forest classifier
# You can customize the parameters as needed (e.g., n_estimators, max_depth, etc.)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model on the scaled training data
rf_classifier.fit(X_train, y_train)
# Make predictions on the scaled test data
y_pred = rf_classifier.predict(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
# Print the classification report
# Testing with real values
XTEST = [1.0, 65.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
XTEST = np.array(XTEST).reshape(1, -1)

 

# Predict with the scaled test input
y_pred_single = rf_classifier.predict(XTEST)
#print("Prediction for single sample:", y_pred_single)
st.title("PREDICTING LUNG CANCER    IN A PATIENT")
 

# Get user inputs
GENDER = st.number_input("Gender (1 for Male, 0 for Female)", min_value=0, max_value=1, step=1, value=1)
AGE = st.number_input("Age", min_value=0, step=1, value=65)
SMOKING = st.number_input("Smoking (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=1)
YELLOW_FINGERS = st.number_input("Yellow Fingers (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=1)
ANXIETY = st.number_input("Anxiety (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=1)
PEER_PRESSURE = st.number_input("Peer Pressure (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=0)
CHRONIC_DISEASE = st.number_input("Chronic Disease (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=0)
FATIGUE = st.number_input("Fatigue (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=1)
ALLERGY = st.number_input("Allergy (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=0)
WHEEZING = st.number_input("Wheezing (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=0)
ALCOHOL_CONSUMING = st.number_input("Alcohol Consuming (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=0)
COUGHING = st.number_input("Coughing (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=0)
SHORTNESS_OF_BREATH = st.number_input("Shortness of Breath (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=0)
SWALLOWING_DIFFICULTY = st.number_input("Swallowing Difficulty (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=0)
CHEST_PAIN = st.number_input("Chest Pain (1 for Yes, 0 for No)", min_value=0, max_value=1, step=1, value=1)

# Create an array from user inputs
XTEST = np.array([GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE,
                  ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]).reshape(1, -1)

 

# Predict with the scaled test input
if st.button("PREDICT"):
    y_pred_single = rf_classifier.predict(XTEST)
    if y_pred_single[0] == 1:
        st.markdown('<p style="color:red; font-size:20px;">PATIENT HAVING LUNG CANCER</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green; font-size:20px;">PATIENT HAVING No LUNG CANCER</p>', unsafe_allow_html=True)
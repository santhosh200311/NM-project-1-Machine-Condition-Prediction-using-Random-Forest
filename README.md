

# Machine Condition Monitoring using Random Forest

**Project by: Santhosh G**
**2nd Year, Mechanical Engineering**
**ARM College of Engineering & Technology**
**Course: Data Analysis in Mechanical Engineering**

---

## Overview

This project focuses on predicting the operating condition of industrial machines using a **Random Forest Classifier**. The model takes in various machine-related parameters such as temperature, vibration levels, oil quality, RPM, and more, and then determines whether the machine is running normally or showing signs of a fault.

As part of learning how data analysis can help in mechanical systems, this project helped me understand how machine learning models are trained, used, and deployed.

---

## Setup Instructions

Before starting, make sure all required Python packages are installed. You can do this by running:

```bash
pip install -r requirements.txt
```

---

## Required Files

These three files are necessary for making predictions:

* `random_forest_model.pkl` – The trained machine learning model.
* `scaler.pkl` – A StandardScaler used to normalize the data during training.
* `selected_features.pkl` – A list of features used to ensure correct column order during prediction.

These files should be in the same directory as your prediction script or properly referenced in the code.

---

## How the Prediction Works

The prediction process happens in four main steps:

1. **Loading Saved Files**

   * Load the trained model and scaler using `joblib`.
   * Load the feature list to make sure inputs are in the correct order.

2. **Creating the Input**

   * Prepare a one-row DataFrame with the same features that were used in training.
   * Feature names must match exactly.

3. **Preprocessing the Input**

   * The input data is scaled using the same method that was used during model training.

4. **Making the Prediction**

   * The model predicts the machine's condition (e.g., Normal or Faulty).
   * It also provides probability scores for each possible condition.

---

## Sample Code for Prediction

Here is a sample template for how to run a prediction using `predict.py`:

```python
import joblib
import pandas as pd

# Load model and preprocessing tools
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Example input data
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Arrange features in correct order
new_data = new_data[selected_features]

# Scale the data
scaled_data = scaler.transform(new_data)

# Predict the condition
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Condition:", prediction[0])
print("Prediction Confidence:", prediction_proba[0])
```

---

## Important Points to Remember

* Input features must exactly match the ones used during model training.
* The order of features is very important; do not rearrange columns.
* Input values should be realistic and within expected operating ranges.

---

## Retraining the Model (Optional)

If needed, the model can be retrained with new data by following the same steps used originally:

* Use consistent preprocessing steps.
* Apply the same feature scaling.
* Save the new model and tools using `joblib`.

---

## Real-World Applications

This kind of system can be used in:

* Factory environments to monitor machine health.
* Preventive maintenance scheduling.
* IoT-based monitoring systems for industrial equipment.

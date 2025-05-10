**Name:** Nandha Kumar S
**Year:** 2nd Year
**Department:** Mechanical Engineering
**Course:** Data Analysis in Mechanical Engineering
**College:** ARM College of Engineering & Technology

---

# Machine Condition Prediction Using Random Forest

This project is focused on predicting the condition of industrial machines using a machine learning model called **Random Forest Classifier**. By analyzing important parameters such as temperature, vibration levels, oil quality, RPM, and other operating features, the model can help identify whether a machine is functioning normally or may be at risk of a fault.

This is part of my course project in *Data Analysis in Mechanical Engineering*.

---

## Getting Started

To run this project successfully, make sure to install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Project Files Overview

This project depends on the following key files:

* **`random_forest_model.pkl`** � This is the trained machine learning model.
* **`scaler.pkl`** � A scaler used to normalize the input data, ensuring that all features are on the same scale.
* **`selected_features.pkl`** � A list of the selected features used during model training. These need to match the input order when making predictions.

Ensure these files are available in your working directory before running predictions.

---

## How the Prediction Works

Here is a simple breakdown of how the prediction process is carried out:

1. **Loading the Model and Tools:**

   * The model, scaler, and selected feature list are loaded using `joblib.load()`.

2. **Preparing the Input:**

   * You need to create a single-row data entry (as a DataFrame) with all the required features.

3. **Preprocessing:**

   * The input data is scaled using the loaded scaler so it matches the format of the training data.

4. **Making the Prediction:**

   * The scaled data is passed to the model for prediction.
   * The model returns both the predicted class (e.g., normal or faulty) and the probability scores.

---

## Example Prediction Script

Here�s an example script (`predict.py`) that shows how to use the model:

```python
import joblib
import pandas as pd

# Load the saved model and tools
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Example input (replace with actual sensor values)
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

# Match feature order
new_data = new_data[selected_features]

# Scale the input data
scaled_data = scaler.transform(new_data)

# Make predictions
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Class:", prediction[0])
print("Prediction Probabilities:", prediction_proba[0])
```

---

## Important Points

* Always provide **exactly the same features** that were used in training.
* Input values must be realistic and within the range of the training data.
* The order of features in the DataFrame must not be changed.

---

## Optional: Retraining the Model

If you need to retrain or update the model:

* Follow the same steps used in the original training pipeline.
* Use consistent data preprocessing and scaling methods.
* Save the new model and tools again using `joblib`.

---

## Real-World Applications

* Predictive maintenance in industrial environments.
* Condition monitoring of machines in manufacturing units.
* Integration with sensor systems for real-time alerts.

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Initialize Flask app
app = Flask(__name__)

# Load and process dataset
file_path = 'cleaned_tsunami_data.csv'  # Replace with your file path
tsunami_data = pd.read_csv(file_path)

# Prepare features (X) and target (y)
X = tsunami_data[['LATITUDE', 'LONGITUDE', 'EQ_MAGNITUDE', 'EQ_DEPTH']]
y = (tsunami_data['TS_INTENSITY'] > 0).astype(int)  # Target: 1 if tsunami, 0 otherwise

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train an SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Route for predicting tsunami occurrence
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from the request
        data = request.json
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        eq_magnitude = float(data['eq_magnitude'])
        eq_depth = float(data['eq_depth'])

        # Standardize the input
        user_input = scaler.transform([[latitude, longitude, eq_magnitude, eq_depth]])

        # Make a prediction
        prediction = model.predict(user_input)[0]
        tsunami_prediction = "Yes" if prediction == 1 else "No"

        # Categorize earthquake intensity based on magnitude and depth
        if 0 <= eq_magnitude < 5 and eq_depth == 100:
            intensity = "Low-intensity earthquake"
        elif 5 <= eq_magnitude < 9 and 100 <= eq_depth < 500:
            intensity = "Medium-intensity earthquake"
        elif 9 <= eq_magnitude <= 11 and eq_depth >= 500:
            intensity = "High-intensity earthquake"
        else:
            intensity = "Unknown intensity"

        # Determine likely cause
        if eq_magnitude >= 7 and eq_depth < 100:
            cause = "Possible Cause: Shallow, high-magnitude earthquake."
        elif eq_depth >= 500:
            cause = "Possible Cause: Deep-sea earthquake or tectonic plate movement."
        elif latitude >= 30 and latitude <= 60:
            cause = "Possible Cause: Underwater landslide or subduction zone activity."
        else:
            cause = "Cause: Other geophysical phenomena."

        # Return the prediction and additional information
        return jsonify({
            "tsunami_prediction": tsunami_prediction,
            "earthquake_intensity": intensity,
            "likely_cause": cause,
            "details": {
                "latitude": latitude,
                "longitude": longitude,
                "eq_magnitude": eq_magnitude,
                "eq_depth": eq_depth
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

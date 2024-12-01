import pandas as pd

# Load the uploaded dataset to examine its structure
file_path = 'cleaned_tsunami_data.csv'
tsunami_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
tsunami_data

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Define features and target variable
X = tsunami_data[['LATITUDE', 'LONGITUDE', 'EQ_MAGNITUDE', 'EQ_DEPTH']]
y = (tsunami_data['TS_INTENSITY'] > 0).astype(int)  # Target: 1 if tsunami, 0 otherwise

# Standardize the data (SVM performs better with scaled data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train an SVM Classifier
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Example prediction for user input
latitude = float(input("Enter latitude: "))
longitude = float(input("Enter longitude: "))
eq_magnitude = float(input("Enter earthquake magnitude: "))
eq_depth = float(input("Enter earthquake depth: "))

# Categorize earthquake intensity based on magnitude and depth
if 0 <= eq_magnitude < 5 and eq_depth == 100:
    print("Low-intensity earthquake")
elif 5 <= eq_magnitude < 9 and 100 <= eq_depth < 500:
    print("Medium-intensity earthquake")
elif 9 <= eq_magnitude <= 11 and eq_depth >= 500:
    print("High-intensity earthquake")

# Determine the likely cause of the tsunami based on user input
if eq_magnitude >= 7 and eq_depth < 100:
    print("Possible Cause: Shallow, high-magnitude earthquake.")
elif eq_depth >= 500:
    print("Possible Cause: Deep-sea earthquake or tectonic plate movement.")
elif latitude >= 30 and latitude <= 60:
    print("Possible Cause: Underwater landslide or subduction zone activity.")
else:
    print("Cause: Other geophysical phenomena.")

# Standardize the user input
user_input = scaler.transform([[latitude, longitude, eq_magnitude, eq_depth]])

# Predict tsunami occurrence
prediction = model.predict(user_input)
print("Tsunami Prediction:", "Yes" if prediction[0] == 1 else "No")

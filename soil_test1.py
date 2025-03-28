import pandas as pd
import joblib

# Load the saved model
model_path = 'soil_health_model.pkl'
rf_classifier = joblib.load(model_path)

# Load the test data (without the 'Output' column)
test_file_path = 'check_soil_data.csv'  # path to the test CSV file
soil_test_data = pd.read_csv(test_file_path)

# Display the number of rows in the test dataset
rowcount = 0
for row in open(test_file_path): 
    rowcount += 1
print("Number of lines present in test file:", (rowcount - 1))

# Preprocess the test data
# Test data contains only the feature columns, no 'Output'
X_test = soil_test_data  # No need to drop 'Output' since it doesn't exist

# Make predictions on the test data
rf_predictions = rf_classifier.predict(X_test)

# Manually map predictions (0 = Non Fertile, 1 = Fertile)
label_mapping = {0: 'Non Fertile', 1: 'Fertile'}
predicted_labels = [label_mapping[pred] for pred in rf_predictions]

print("Predicted Labels:")
print(predicted_labels)
import pandas as pd
import joblib


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


if 'Output' in soil_test_data.columns:
    X_test = soil_test_data.drop(columns=['Output'])
else:
    X_test = soil_test_data

# Make predictions on the test data
rf_predictions = rf_classifier.predict(X_test)

# Manually map predictions (1 = Non Fertile, 0 = Fertile)
label_mapping = {1: 'Non Fertile', 0: 'Fertile'}
predicted_labels = [label_mapping[pred] for pred in rf_predictions]

print("Predicted Labels:")
print(predicted_labels)

# Add predictions to the original dataframe
soil_test_data['Predicted_Output'] = predicted_labels

non_fertile_data = soil_test_data[soil_test_data['Predicted_Output'] == 'Non Fertile']

# Export the non-fertile data to a new CSV file
output_file_path = 'non_fertile_soil_data.csv'
non_fertile_data.to_csv(output_file_path, index=False)

print(f"Non-fertile soil data exported to {output_file_path}")
print(f"Number of non-fertile soil samples: {len(non_fertile_data)}")

# Load the non-fertile soil data
non_fertile_data = pd.read_csv('non_fertile_soil_data.csv')

# Load the prevention data
prevent_data = pd.read_csv('prevent.csv')

def extract_range_values(range_string):
    """
    Splits a range string like '20-50' or '0.2-0.6' into two boundary values (as float or int).
    Returns a list of the boundary values.
    Handles cases where the range string does not contain a hyphen.
    """
    try:
        # Split the range string by '-'
        if '-' in range_string:
            min_val, max_val = range_string.split('-')
            # Convert to float if decimal, else int
            min_val = float(min_val.strip()) if '.' in min_val else int(min_val.strip())
            max_val = float(max_val.strip()) if '.' in max_val else int(max_val.strip())
            return [min_val, max_val]
        else:
            # Handle cases without a hyphen (e.g., a single value)
            # You might want to return a single value or handle it differently based on your needs
            # Here, I'm assuming it's a single value and returning it as both min and max
            single_val = float(range_string.strip()) if '.' in range_string else int(range_string.strip())
            return [single_val, single_val]

    except ValueError:
        # In case of unexpected format, return an empty list or handle the error
        print(f"Invalid range format: {range_string}")
        # Return (None, None) instead of an empty list
        return None, None
    

# Function to convert threshold strings to numeric ranges
def extract_range(threshold_str):
    try:
        numbers = [float(x) for x in threshold_str.replace(',', '').split() if x.replace('.', '').isdigit()]
        return min(numbers), max(numbers)
    except:
        return None, None
    

# Function to check if a value is within a range
def is_within_range(value, range_min, range_max):
    return range_min <= value <= range_max if range_min is not None and range_max is not None else False


# List of all properties to check
properties = ['pH', 'EC', 'OC', 'OM', 'N', 'P', 'K', 'Zn', 'Fe', 'Cu', 'Mn', 'Sand', 'Silt', 'Clay', 'CaCO3', 'CEC', 'Temperature', 'Humidity']

# Prepare a list to store results
results = []

# Process each row in the non-fertile soil data
for index, soil_row in non_fertile_data.iterrows():
    for property_name in properties:
        if property_name in soil_row.index and property_name in prevent_data['Property'].values:
            value = soil_row[property_name]
            prevent_row = prevent_data[prevent_data['Property'] == property_name].iloc[0]
            range_min, range_max = extract_range_values(prevent_row['Threshold Value (Typical Range)'])

            if not is_within_range(value, range_min, range_max):
                results.append({
                    'Soil Sample ID': index + 1,
                    'Property': property_name,
                    'Value': value,
                    'Threshold Range': prevent_row['Threshold Value (Typical Range)'],
                    'Organic Prevention Methods': prevent_row['Organic Prevention Methods'],
                    'Commercial Products': prevent_row['Commercial Products'],
                    'Remarks': prevent_row['Remarks']
                })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv('soil_analysis_result.csv', index=False)

print("Analysis complete. Results have been saved to 'soil_analysis_results.csv'.")
print(f"Total out-of-range properties found: {len(results_df)}")
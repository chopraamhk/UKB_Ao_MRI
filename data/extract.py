##the following code is to extract values from column 1 which has 20210-2 value in column 2, i.e., extracting only images in instance 1 (1st image instance)

import pandas as pd

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('ukb674295.csv')  # Replace 'your_input_file.csv' with the actual file path

# Filter rows where column 2 contains '20210_2'
filtered_data = data[data['20210-2.0'].str.contains('20210_2', na=False)]

# Extract values from column 1 of the filtered data
values_in_column1 = filtered_data['eid']

# Create a new DataFrame with the extracted values
output_data = pd.DataFrame({'eid': values_in_column1})

# Write the output DataFrame to a CSV file
output_data.to_csv('instance1.csv', index=False)  # Replace 'output_values.csv' with your desired output file name

# pandas is a library used for data manipulation and analysis.
# display from IPython.display is used to display dataframes in a readable format in Jupyter notebooks
import pandas as pd
from IPython.display import display

# pd.read_csv('class_grades_2c.csv') reads the CSV file named class_grades_2c.csv into a DataFrame called df.
df = pd.read_csv('first.csv')

# display(df.head()) shows the first few rows of the DataFrame to give an overview of the data
display(df.head())

# selects only the columns with numeric data types from the DataFrame and stores them in numeric_df.
numeric_df = df.select_dtypes(include='number')

# Display the first few rows of numeric columns
display(numeric_df.head())

# Calculate summary statistics
averages = numeric_df.mean().to_dict()
max_values = numeric_df.max().to_dict()
min_values = numeric_df.min().to_dict()

# This dictionary summary stores the calculated averages, maximum values, and minimum values
summary = {
    'Averages': averages,
    'Maximum Values': max_values,
    'Minimum Values': min_values
}

# This loop iterates over the summary dictionary.
# For each key (e.g., 'Averages', 'Maximum Values', 'Minimum Values'), it prints the key.
# pd.DataFrame(value.items(), columns=["Metric", key]) creates a DataFrame from the dictionary items and displays it in a readable format.

for key, value in summary.items():
    print(f"\n{key}:")
    display(pd.DataFrame(value.items(), columns=["Metric", key]))

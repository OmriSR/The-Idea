import bisect
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from brca.utils import load_and_split_data
from my_transformers import TableToEmbedding

def find_optimal_num_of_components(n_components=12,th=0.85):
    X,y = load_and_split_data(file_path="Titanic/Titanic.csv",target_column="Survived")

    table_to_embedding = TableToEmbedding(
        expended_col_names_mapper=None,
        prefix=None,
        use_textual_numbers=None
    )
    pca = PCA()

    X_train_emedded = table_to_embedding.transform(X)

    pca.fit(X_train_emedded)
    
    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    # Compute cumulative explained variance
    cumulative_variance=np.cumsum(explained_variance_ratio)

    print(cumulative_variance)
    print(bisect.bisect_left(cumulative_variance, th))

def plot_cumulative_explained_variance(post_fit_pca: PCA):
    # Get the explained variance ratio
    explained_variance_ratio = post_fit_pca.explained_variance_ratio_

    # Compute cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Plot the cumulative explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')

    plt.title('Cumulative Explained Variance vs Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.show()



# Function to extract model details and format into a DataFrame
def extract_model_details(data_section, dataset_name):
    rows = []
    for section in data_section:
        for formatting_method, models in section.items():
            for model in models:
                model_details = json.loads(model)
                row = {
                    'Dataset': dataset_name,
                    'Formatting Method': formatting_method,
                    'Model Params': model_details['model params'],
                    'Num of Components': model_details['num of component'],
                    'Best Training Score': model_details['best training score'],
                    'Accuracy': model_details['accurcy'],
                    'F1 Score': model_details['f1']
                }
                rows.append(row)
    return rows

def json_to_excel(json_string, excel_filename):
    """
    Convert a given JSON string into an Excel file with specific formatting.

    Args:
    json_string (str): A JSON string containing datasets and formatting methods.
    excel_filename (str): The name of the Excel file to be generated (including .xlsx extension).
    
    Returns:
    None: The function saves the Excel file in the given location.
    """
    # Load the JSON string into a Python dictionary
    data = json.loads(json_string)

    # Extract rows for both "New Titanic" and "Titanic"
    new_titanic_rows = extract_model_details(data['New Titanic'], 'New Titanic')
    # titanic_rows = extract_model_details(data['Titanic'], 'Titanic')

    # Create a DataFrame for both datasets
    df_combined = pd.DataFrame(new_titanic_rows)

    # Export the DataFrame to an Excel file
    df_combined.to_excel(excel_filename, index=False)
import json
from typing import List
from numpy import ndarray
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

def row_2_enbedding(X:pd.DataFrame, transform_to_str_list_func) -> ndarray :
    """
    Showcases the 'zero-touch' or 'json' formatting, 
    'zero-touch' = each row is transformed into a string consisting of only its values separated by commas,
    'json' = each row is transformed into a string consisting of col-name and value
    and then encoded to embedding.
    """
    sentence_transformer = SentenceTransformer(MODEL)

    string_rows = transform_to_str_list_func(X)
    return sentence_transformer.encode(string_rows)

###### row to string methods ######
def transform_to_json(X: pd.DataFrame) -> List[str]:
    """
    generic function to create json formatting out of whole data frame
    """
    return [
            json.dumps({col: str(value) for col, value in row.items()})
            for _, row in X.iterrows()
        ]

def zero_touch_formatting(X:pd.DataFrame) -> List[str]:
    """
    Transform each row in dataframe to a string of its values
    """
    return X.apply(lambda row: ','.join(row.astype(str)), axis=1).tolist()


##### Titanic specific ######
def titanic_passanger_to_json(row:pd.Series) -> dict:
    # Structured JSON-like format
    structured_format = {
        "PassengerId": row['PassengerId'],
        "Name": row['Name'],
        "Age": row['Age'],
        "Sex": row['Sex'],
        "Pclass": row['Pclass'],
        "Fare": row['Fare'],
        "SibSp": row['SibSp'],
        "Parch": row['Parch'],
        "Cabin": row['Cabin'],
        "Embarked": row['Embarked']
    }
    return structured_format

def describe_titanic_passenger(row:pd.Series) -> str:
    """
    Takes a dictionary representing a row of Titanic data and returns a natural language description.

    Args:
    row (dict): A dictionary with keys ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'].

    Returns:
    str: A natural language description of the passenger.
    """
    description = f"Passenger {row['PassengerId']}, named {row['Name']}, is a {row['Age']}-year-old {row['Sex']}."
    
    # Adding class and fare information
    description += f" They traveled in class {row['Pclass']} with a fare of ${row['Fare']}."

    # Adding sibling/spouse and parent/child information
    if row['SibSp'] > 0:
        description += f" They were accompanied by {row['SibSp']} sibling(s) or spouse."
    if row['Parch'] > 0:
        description += f" They also traveled with {row['Parch']} parent(s) or child(ren)."

    # Adding cabin and embarkation details if available
    if row['Cabin']:
        description += f" They stayed in cabin {row['Cabin']}."
    if row['Embarked']:
        description += f" They embarked from {row['Embarked']}."

    return description

def natural_language_description_formatting(X:pd.DataFrame) -> List[str]:
    """
    Transform each row in dataframe to a string holding Natural Language description
    """
    return X.apply(describe_titanic_passenger, axis=1).tolist()

def create_hybrid_pass_description(row:pd.Series) -> str:
    return f"{transform_to_json(row)},{describe_titanic_passenger(row)}"

def json_nld_hybrid_formatting(X:pd.DataFrame):
    """
    Transform each row in dataframe to both:
    * Json 
    * string holding Natural Language description
    """
    return X.apply(describe_titanic_passenger, axis=1).tolist()


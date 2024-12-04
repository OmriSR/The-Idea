import sys
import os
import pandas as pd

from ucimlrepo import fetch_ucirepo 

from sklearn import svm
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.metrics import classification_report


# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the Thesis directory
thesis_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Add the Thesis directory to the system path
sys.path.append(thesis_dir)

from my_transformers import TableToEmbedding

RECORD_EXTRA_INFO_PREFIX = ("medical data collected for the purpose of classifying hepatitis patients into survival and non-survival categories."
                            "These attributes are both categorical and numerical, providing a diverse set of features for analysis.")

def get_hepatitis_X_y(describe_data:bool = False) -> tuple[pd.DataFrame,pd.Series]:
    # fetch dataset 
    hepatitis = fetch_ucirepo(id=46) 

    # data (as pandas dataframes) 
    X = hepatitis.data.features 
    y = hepatitis.data.targets 
    
    if describe_data:
        # metadata 
        print(hepatitis.metadata) 
        # variable information 
        print(hepatitis.variables) 

    return X,y.values.ravel()


def run_classification_according_to_params(add_prefix:bool, separate_encodding:bool, use_textual_numbers:bool):
    X,y = get_X_and_y()

    # if both values are true we will encode the prefix spearatly
    encode_together = add_prefix and separate_encodding is False 

    # transform tabular dataset to embeddings metrix
    table_to_embedding = TableToEmbedding(expended_col_names_mapper=None,
                                          prefix= RECORD_EXTRA_INFO_PREFIX if encode_together else None, 
                                          use_textual_numbers=use_textual_numbers)
    
    X_embeddings = table_to_embedding.transform(X)

    if separate_encodding:
        # enocode only the Prefix 
        prefix_embedding = table_to_embedding.sentence_transformer.encode(RECORD_EXTRA_INFO_PREFIX)

        # for each emcedding, concatenate the embedded prefix
        X_embeddings = [prefix_embedding + embdd for embdd in X_embeddings]

    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.15, random_state=42)

    param_grid = {
        "C": [0.1,1,10,100,],
        "gamma": [1,0.1,0.01,0.001,],
        "kernel": ["rbf","poly","sigmoid",],
    }

    svm_model = svm.SVC()
    grid_search = GridSearchCV(
        svm_model,
        param_grid,
        cv=5,
        verbose=0,
        n_jobs=1,
    )

    grid_search.fit(X_train, y_train)

    print(
        "Best parameters found: ",
        grid_search.best_params_,
    )
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    
    print(
        f"###### with {'' if use_textual_numbers  else 'out'} textual numbers (1->'one') and"
        f" with {'' if add_prefix else 'out'} prefix ######"
    )   
    if separate_encodding:
        print("The prefix was encoded separatly and then added")

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    relevant_boolean_combos = [[True,True,True], [False,False,True], [True,False,True], [False,False,False]]
    for state in relevant_boolean_combos:
        run_classification_according_to_params(*state)

    # the reported resault using SVM on this data is 82
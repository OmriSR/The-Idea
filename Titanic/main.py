import json
import os
import sys
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the Thesis directory
thesis_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Add the Thesis directory to the system path
sys.path.append(thesis_dir)
from Titanic.utils import json_to_excel
from compare_pipeline.formating_utils import json_nld_hybrid_formatting, natural_language_description_formatting, row_2_enbedding, transform_to_json, zero_touch_formatting
from compare_pipeline.main import create_result_json, generic_flow, print_score, svm_gridsearch,load_and_split_data
from my_transformers import MODEL, TableToEmbedding

def main(add_informative_prefix:str,use_textual_numbers:bool):
    X,y = load_and_split_data(file_path="Titanic/Titanic.csv",target_column="Survived")

    table_to_embedding = TableToEmbedding(
        expended_col_names_mapper=None,
        prefix=add_informative_prefix,
        use_textual_numbers=use_textual_numbers
    )
    pca_12_components = PCA(n_components=12)
    pca_24_components = PCA(n_components=24)

    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42,)

    X_train_emedded = table_to_embedding.transform(X_train)

    for pca, use_pca in [(pca_12_components,True),(pca_24_components,True), (None,False)]:
        # reduce the dimensions of each embedding to better match the original number of features
        if use_pca:
            X_train_reduced = pca.fit_transform(X_train_emedded)

        param_grid = {
            "C": [0.1,1,10,100,],
            "gamma": [1,0.1,0.01,0.001],
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

        grid_search.fit(X_train_reduced, y_train)

        X_test_embedded = table_to_embedding.transform(X_test)

        if use_pca:
            X_test_reduced = pca.transform(X_test_embedded)

        print(
            f"Best parameters found with: ",
            grid_search.best_params_,
        )
        best_model = grid_search.best_estimator_

        # 5-fold cross-validation
        cv_scores = cross_val_score(best_model, X_train_reduced, y, cv=5)  

        y_pred = best_model.predict(X_test_reduced)

        print(f"use_textual_numbers: {use_textual_numbers}, add_informative_prefix: {add_informative_prefix}, use_pca: {use_pca}")
        print(classification_report(y_test, y_pred))

        print(cv_scores)

def preprocess_data(X: pd.DataFrame):
    """
    Transform samples from tabular represntaion to a textual Json and then to embedding
    """
    sentence_transformer = SentenceTransformer(MODEL)
    
    X_as_json = [json.dumps(
            {
                col : str(value) for col, value in row.items()
            }
        ) for _, row in X.iterrows()]

    embedding = sentence_transformer.encode(X_as_json)
    return embedding

def full_titanic_flow(transform_to_str_list_func):
    """
    This function runs (row->str wise) flow
    using SVM and 3 versions of PCA usage 
    """

    X_train,X_test = load_and_split_data(file_path="Titanic/train.csv",target_column="Survived")

    y_train,y_test= load_and_split_data(file_path="Titanic/test.csv",target_column="Survived")

    number_of_features = X_train.shape[1]

    X_embeding_train = row_2_enbedding(X_train,transform_to_str_list_func)
    res_list = []

    for comp in [None, number_of_features,number_of_features*2]:
        if comp is not None:
            pca = PCA(n_components=comp)
            X_reduced_train = pca.fit_transform(X_embeding_train)
        else:
            X_reduced_train = X_embeding_train

        grid_search = svm_gridsearch(X_reduced_train,y_train)
        best_model = grid_search.best_estimator_

        # evaluation
        X_test_embedding = row_2_enbedding(X_test, transform_to_str_list_func)

        X_reduced_test = X_test_embedding if comp is None else pca.transform(X_test_embedding)

        test_cv_scores  = cross_val_score(best_model, X_reduced_test, y_test, cv=5, scoring='f1')  

        # print_score(grid_search=grid_search,test_cv_scores=test_cv_scores,comp=comp)
        res_list.append(create_result_json(grid_search=grid_search,test_cv_scores=test_cv_scores,comp=comp))
        
    return res_list

if __name__ == "__main__":
    new_titanic_X, new_titanic_y = load_and_split_data(file_path="Titanic/train.csv",target_column="Survived")
    # titanic_X, titanic_y = load_and_split_data(file_path="Titanic/Titanic.csv",target_column="Survived")
    
    new_titanic_data = (new_titanic_X, new_titanic_y, "New Titanic")
    # titanic_data = (titanic_X, titanic_y,"Titanic")

    results_sum = {}
    for X,y,dataset_name in [new_titanic_data]:
        results_sum[dataset_name] = list()
        for formating_method,format in [(transform_to_json,'Json'), (natural_language_description_formatting,'Natural Language Description'),(json_nld_hybrid_formatting, "Hybrid")]:
            cur_res = generic_flow(X,y,formating_method)
            results_sum[dataset_name].append({format:cur_res})
    
    json_res = json.dumps(results_sum)
    with open("Random Forest - NLD & Json Formatting 1.0.txt", 'w') as file:
        file.write(json_res)
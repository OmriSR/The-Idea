import json
import os
from scipy.stats import randint
import sys
from typing import List

from numpy import mean
import pandas as pd

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from xgboost import XGBClassifier


# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the Thesis directory
thesis_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Add the Thesis directory to the system path
sys.path.append(thesis_dir)
# from compare_pipeline.zero_touch import zero_touch_formatting
from compare_pipeline.formating_utils import row_2_enbedding, zero_touch_formatting, transform_to_json
from hepatitis.utils import get_hepatitis_X_y

def load_and_split_data(file_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(file_path)
    y = df[target_column]
    X = df.drop(
        labels=[target_column],
        axis=1,
    )
    return X,y

def xgboost_gridsearch(X,y):
    param_grid = {
        'n_estimators': [100, 200],          # Fewer values to test
        'max_depth': [3, 5],                # Focus on typical tree depths
        'learning_rate': [0.1, 0.3],        # Common learning rates
        'subsample': [0.8, 1.0],            # Slightly reduce subsample options
        'colsample_bytree': [0.8],          # Fix this to a commonly effective value
        'gamma': [0, 0.1],                  # Reduced options for minimum split loss
        'reg_alpha': [0],                   # Often less impactful, fixed to 0
        'reg_lambda': [1]                   # Fixed to the default value
    }

    xgb_model = XGBClassifier(eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                            cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
    
    grid_search.fit(X, y)
    return grid_search

def rand_forest_gridsearch(X,y):
    param_grid = {
    'n_estimators': randint(50, 300),              # Number of trees
    'max_depth': randint(3, 20),                   # Depth of trees
    'min_samples_split': randint(2, 10),           # Minimum number of samples for a split
    'min_samples_leaf': randint(1, 5),             # Minimum samples in a leaf
    'max_features':['sqrt', 'log2', None],         # Number of features to consider for splits
    'bootstrap': [True, False]                     # Whether to use bootstrapping
    }

    rf_model = RandomForestClassifier()
    random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid,
                                   n_iter=50, cv=3, random_state=42, n_jobs=-1, verbose=1, scoring='accuracy')    
    random_search.fit(X, y)
    return random_search

def svm_gridsearch(X,y):
    param_grid = {
        "C": [0.1,1,10],
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

    grid_search.fit(X, y)
    return grid_search

def print_score(grid_search:GridSearchCV, acc_test_cv_scores:List[str], f1_test_cv_scores:List[str], comp:int):
    print("________________________________________________________")
    print(f"model params: {grid_search.best_params_}, num of component:{comp}\n")
    print(f"best training score: {grid_search.best_score_}\n")
    print(f"accurcy : {acc_test_cv_scores}. avg test res: {mean(acc_test_cv_scores)}")
    print(f"F1 : {f1_test_cv_scores}. avg test res: {mean(f1_test_cv_scores)}")
    print("________________________________________________________\n")

def create_result_json(grid_search:GridSearchCV, acc_test_cv_scores:List[str], f1_test_cv_scores:List[str], comp:int):
    return json.dumps(
        {
            "model params": grid_search.best_params_,
            "num of component":comp,
            "best training score":grid_search.best_score_,
            "accurcy":mean(acc_test_cv_scores),
            "f1":mean(f1_test_cv_scores),
        }
    )
    

def generic_flow(X:pd.DataFrame, y:pd.Series, transform_to_str_list_func):

    number_of_features = X.shape[1]

    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

    X_embeding_train = row_2_enbedding(X_train,transform_to_str_list_func)
    res_list = []

    for comp in [None, number_of_features,number_of_features*2]:
        if comp is not None:
            pca = PCA(n_components=comp)
            X_reduced_train = pca.fit_transform(X_embeding_train)
        else:
            X_reduced_train = X_embeding_train

        grid_search = rand_forest_gridsearch(X_reduced_train,y_train)
        # grid_search = xgboost_gridsearch(X_reduced_train,y_train)
        # grid_search = svm_gridsearch(X_reduced_train,y_train)
        best_model = grid_search.best_estimator_
        best_model_copy = grid_search.best_estimator_

        # evaluation
        X_test_embedding = row_2_enbedding(X_test, transform_to_str_list_func)

        X_reduced_test = X_test_embedding if comp is None else pca.transform(X_test_embedding)

        f1_test_cv_scores  = cross_val_score(best_model, X_reduced_test, y_test, cv=5, scoring='f1')  
        acc_test_cv_scores  = cross_val_score(best_model_copy, X_reduced_test, y_test, cv=5)  

        # print_score(grid_search=grid_search,f1_test_cv_scores=f1_test_cv_scores,acc_test_cv_scores=acc_test_cv_scores,comp=comp)
        res_list.append(create_result_json(grid_search=grid_search,f1_test_cv_scores=f1_test_cv_scores,acc_test_cv_scores=acc_test_cv_scores,comp=comp))

    return res_list

if __name__ == "__main__":
    titanic_X, titanic_y = load_and_split_data(file_path="Titanic/train.csv",target_column="Survived")
    brca_X, brca_y = load_and_split_data(file_path="brca\\brca.csv",target_column="y")
    brca_y = brca_y.map({'B': 0, 'M': 1})
    hepatitis_X, hepatitis_y = get_hepatitis_X_y()

    titanic_data = (titanic_X, titanic_y,"Titanic")
    brca_data = (brca_X, brca_y,"BRCA")
    hepatitis_data = (hepatitis_X, hepatitis_y,"Hepatitis")
    
    
    for X,y,dataset_name in [titanic_data,brca_data,hepatitis_data]:
        print(f"{dataset_name}:")
        for formating_method,format in [(zero_touch_formatting,"Zero-Touch"),(transform_to_json,'Json')]:
            print(f"{format}:")
            generic_flow(X,y,formating_method)
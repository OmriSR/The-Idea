import json
from logging import INFO, Logger
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from compare_pipeline.main import create_result_json, rand_forest_gridsearch, svm_gridsearch, xgboost_gridsearch

IS_EMBEDDING_MATRIX_CREATED = False
NUM_OF_FEATURES = 11

def create_taitanic_embedding_matrix():
    # Load your sentences from the text file
    with open("Titanic\\chat_gpt_formatter\\titanic_descriptions_no_survived.txt", "r") as file:
        sentences = file.readlines()

    # Initialize the model for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(sentences)

    embedding_matrix = np.array(embeddings)

    # Save the embedding matrix to a file if needed
    np.save("titanic_embedding_matrix.npy", embedding_matrix)

    print("Embedding matrix shape:", embedding_matrix.shape)

def main_flow(cur_model_gridsearch, logger:Logger):
    """
    The data is recieved as text file of strings, then:
    1. creates embedding matrix
    2. dim reduction
    3. finds best params and trains model
    4. cross-validation
    """
    with open("titanic_embedding_matrix.npy", "rb") as matrix:
        embedding_matrix = np.load(matrix)
        y = pd.read_csv("Titanic\\train.csv")["Survived"]
        logger.info(f"Loaded matrix and target column with shapes {embedding_matrix.shape} , {y.shape}")
        different_dims_results = []

        for comp in [None, NUM_OF_FEATURES,NUM_OF_FEATURES*2]:
            logger.info(f"starting {comp} component reduction run")
            if comp is not None:
                pca = PCA(n_components=comp)
                embedding_matrix_reduced = pca.fit_transform(embedding_matrix)
                logger.info(f"new reduced matrix shape {embedding_matrix_reduced.shape}")
            else:
                embedding_matrix_reduced = embedding_matrix

            X_train, X_test, y_train, y_test = train_test_split(embedding_matrix_reduced,y,test_size=0.2,random_state=42)

            grid_search = cur_model_gridsearch(X_train,y_train)
            best_model = grid_search.best_estimator_
            best_model_copy = grid_search.best_estimator_

            # evaluation
            logger.info(f"Evaluating the best model with crossvalidation")
            f1_test_cv_scores  = cross_val_score(best_model, X_test, y_test, cv=5, scoring='f1')  
            acc_test_cv_scores  = cross_val_score(best_model_copy, X_test, y_test, cv=5)  

            different_dims_results.append(create_result_json(grid_search=grid_search,f1_test_cv_scores=f1_test_cv_scores,acc_test_cv_scores=acc_test_cv_scores,comp=comp))

    return different_dims_results

def chat_gpt_formatter(create_new_embedding_metrix:bool, logger:Logger):
    if create_new_embedding_metrix:
        logger.info("Creating embedding matrix from Natural Language Description (ChatGPT origin)")
        create_taitanic_embedding_matrix()

    for model_func, model_name in [(rand_forest_gridsearch,"rand_forest"),(xgboost_gridsearch,"XGBoost"),(svm_gridsearch,"SVM")]:
        logger.info(f"starting {model_name} evaluation")
        different_dims_results = main_flow(model_func,logger=logger)
        json_res = json.dumps(different_dims_results)
        with open(f"{model_name} - chatGPT formatting.txt", 'w') as file:
            file.write(json_res)

if __name__ == "__main__":
    chat_gpt_formatter(create_new_embedding_metrix=IS_EMBEDDING_MATRIX_CREATED,logger=Logger(name="thesis_log",level=INFO))
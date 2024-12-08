import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import sys
from sklearn.model_selection import cross_val_score

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the Thesis directory
thesis_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Add the Thesis directory to the system path
sys.path.append(thesis_dir)
from compare_pipeline.formating_utils import row_2_enbedding,transform_to_json
from compare_pipeline.main import load_and_split_data
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA


# Function to run KNN with different k and distance methods
def run_knn_with_multiple_k_and_metrics( ks=[3, 5, 7, 9], metrics=["euclidean", "cosine"]):
    results = []
    new_titanic_X, new_titanic_y = load_and_split_data(file_path="Titanic/train.csv",target_column="Survived")
    X_embeding = row_2_enbedding(new_titanic_X,transform_to_json)
    number_of_features = new_titanic_X.shape[1]

    for comp in [None, number_of_features,number_of_features*2]:
        if comp is not None:
            pca = PCA(n_components=comp)
            X_reduced = pca.fit_transform(X_embeding)
        else:
            X_reduced = X_embeding

        # X_train,X_test,y_train,y_test= train_test_split(X_reduced,new_titanic_y,test_size=0.2,random_state=42)

        for k in ks:
            for metric in metrics:
                    print(f"\nRunning KNN with k={k} , comp_num={comp} and metric={metric}")
                    
                    # Initialize KNN with the specified number of neighbors and distance metric
                    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
                    # Calculate accuracy 
                    acc_cv_scores = cross_val_score(knn, X_reduced, new_titanic_y, cv=5, scoring='accuracy')
                    f1_cv_scores = cross_val_score(knn, X_reduced, new_titanic_y, cv=5, scoring='f1')


                    # Fit the KNN model on the training data
                    # knn.fit(X_train, y_train)
                    # # Calac training score to detect overfittings
                    # y_pred = knn.predict(X_test)
                    # test_accuracy = accuracy_score(y_test, y_pred)
                    # test_f1 = f1_score(y_test, y_pred)

                    # Output results
                    print(f"Accuracy for k={k}, metric={metric}: {acc_cv_scores.mean():.4f}")
                    
                    results.append({
                        "k": k,
                        "metric": metric,
                        "comp":comp,
                        # "test_accuracy": test_accuracy,
                        "f1_cv": f1_cv_scores.mean(),
                        "accuracy_cv": acc_cv_scores.mean(),
                    })
    
    return results

if __name__ == "__main__":
    results = run_knn_with_multiple_k_and_metrics(ks=[3, 5, 7, 9], metrics=["euclidean", "cosine"])

    # Print out the results for each combination of k and metric

    json_res = json.dumps(results)
    with open("KNN Json Formatting only cross-validation 1.2.txt", 'w') as file:
        file.write(json_res)

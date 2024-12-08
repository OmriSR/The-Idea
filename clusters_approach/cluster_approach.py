import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import sys

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the Thesis directory
thesis_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Add the Thesis directory to the system path
sys.path.append(thesis_dir)
from compare_pipeline.formating_utils import row_2_enbedding, transform_to_json
from compare_pipeline.main import load_and_split_data

def compare_centroid_classification():
    new_titanic_X, new_titanic_y = load_and_split_data(file_path="Titanic/train.csv",target_column="Survived")
    X_embeding = row_2_enbedding(new_titanic_X,transform_to_json)

    number_of_features = new_titanic_X.shape[1]
    results = []
    for method in ["pca", "t-SNE"]:
        for comp in [None, number_of_features,number_of_features*2]:
            if comp is not None:
                X_reduced = reduce_embedding(X_embeding,comp,method)
            else:
                X_reduced = X_embeding

            X_train,X_test,y_train,y_test= train_test_split(X_reduced,new_titanic_y,test_size=0.2,random_state=42)

            cluster_0 = X_train[y_train == 0]
            cluster_1 = X_train[y_train == 1]

            # Compute centroids
            centroid_0 = np.median(cluster_0, axis=0)
            centroid_1 = np.median(cluster_1, axis=0)

            def compare_vectors(vector):
                # Compute dot products with vector_a and vector_b
                dot_product_0 = np.dot(vector, centroid_0)
                dot_product_1 = np.dot(vector, centroid_1)
            
                # Return 1 if dot product with vector_1 is larger, else return 0
                return int(dot_product_1 > dot_product_0)

            y_pred = list(map(compare_vectors, X_test))
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)

            results.append({
                            "method": method,
                            "comp":comp,
                            "f1": test_f1,
                            "accuracy": test_accuracy
                        })
    return results
    
def reduce_embedding(X,comp_num,method="pca"):
    if method == "pca":
        pca = PCA(n_components=comp_num)
        return pca.fit_transform(X)
    else:
        tsne = TSNE(n_components=comp_num, method='exact', random_state=42)
        return tsne.fit_transform(X)

def reduce_and_plot_embedding(X,y,method="pca"):
    X_reduced = reduce_embedding(X,2,method)
    plot_2dim_binary_labeled_vectors(X_reduced,y)


def plot_2dim_binary_labeled_vectors(X_reduced,y):
    # Scatter plot
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        plt.scatter(
            X_reduced[y == label, 0], 
            X_reduced[y == label, 1], 
            label="survived" if label else "didn't survived"
        )

    # Calculate centroids
    centroids_mean = []
    centroids_median = []
    for label in np.unique(y):
        cluster_points = X_reduced[y == label]
        centroid_mean = np.mean(cluster_points, axis=0)
        centroid_median = np.median(cluster_points, axis=0)
        centroids_mean.append(centroid_mean)
        centroids_median.append(centroid_median)
        
        # Plot mean centroid
        plt.scatter(
            centroid_mean[0], centroid_mean[1], 
            color='red' if label else 'yellow', marker='x', s=200, 
            label="survived mean" if label else "didn't survived mean"
        )
        # Plot median centroid
        plt.scatter(
            centroid_median[0], centroid_median[1], 
            color='blue' if label else 'green', marker='o', s=150, 
            label="survived median" if label else "didn't survived median"
        )

    plt.title("embedding distribution")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # new_titanic_X, new_titanic_y = load_and_split_data(file_path="Titanic/train.csv",target_column="Survived")
    # X_embeding = row_2_enbedding(new_titanic_X,transform_to_json)
    # reduce_and_plot_embedding(X_embeding,new_titanic_y, method="pca")
    res = compare_centroid_classification()
    json_res = json.dumps(res)
    with open("cluster method Json Formatting only cross-validation 1.2.txt", 'w') as file:
        file.write(json_res)

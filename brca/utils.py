import pandas as pd
from sklearn import svm
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from my_transformers import TableToEmbedding

RECORD_EXTRA_INFO_PREFIX = (
    "this represents a patient's specific measurements and characteristics derived from"
    " breast mass images, aiding in the assessment of whether the corresponding mass is"
    " malignant or benign"
)

COLUMN_NAME_TO_EXPANDED_NAME_MAPPER = {
    "x.radius_mean": "Mean radius of the tumor cells",
    "x.texture_mean": "Mean texture of the tumor cells",
    "x.perimeter_mean": "Mean perimeter of the tumor cells",
    "x.area_mean": "Mean area of the tumor cells",
    "x.smoothness_mean": "Mean smoothness of the tumor cells",
    "x.compactness_mean": "Mean compactness of the tumor cells",
    "x.concavity_mean": "Mean concavity of the tumor cells",
    "x.concave_pts_mean": (
        "Mean number of concave portions of the contour of the tumor cells"
    ),
    "x.symmetry_mean": "Mean symmetry of the tumor cells",
    "x.fractal_dim_mean": 'Mean "coastline approximation" of the tumor cells',
    "x.radius_se": "Standard error of the radius of the tumor cells",
    "x.texture_se": "Standard error of the texture of the tumor cells",
    "x.perimeter_se": "Standard error of the perimeter of the tumor cells",
    "x.area_se": "Standard error of the area of the tumor cells",
    "x.smoothness_se": "Standard error of the smoothness of the tumor cells",
    "x.compactness_se": "Standard error of the compactness of the tumor cells",
    "x.concavity_se": "Standard error of the concavity of the tumor cells",
    "x.concave_pts_se": (
        "Standard error of the number of concave portions of the contour of the tumor"
        " cells"
    ),
    "x.symmetry_se": "Standard error of the symmetry of the tumor cells",
    "x.fractal_dim_se": (
        'Standard error of the "coastline approximation" of the tumor cells'
    ),
    "x.radius_worst": "Worst (largest) radius of the tumor cells",
    "x.texture_worst": "Worst (most severe) texture of the tumor cells",
    "x.perimeter_worst": "Worst (largest) perimeter of the tumor cells",
    "x.area_worst": "Worst (largest) area of the tumor cells",
    "x.smoothness_worst": "Worst (most severe) smoothness of the tumor cells",
    "x.compactness_worst": "Worst (most severe) compactness of the tumor cells",
    "x.concavity_worst": "Worst (most severe) concavity of the tumor cells",
    "x.concave_pts_worst": (
        "Worst (most severe) number of concave portions of the contour of the tumor"
        " cells"
    ),
    "x.symmetry_worst": "Worst (most severe) symmetry of the tumor cells",
    "x.fractal_dim_worst": (
        'Worst (most severe) "coastline approximation" of the tumor cells'
    ),
}

def run_separated_encoding_for_prefix_flow(use_expended_col_names:bool = False, 
                                          use_textual_numbers:bool = False
                                        ):
    X,y = load_and_split_data("brca\\brca.csv","y")
    X.drop(labels= "Unnamed: 0",axis=1,inplace=True)
    
    label_encoder = LabelEncoder()
    # Encoding the target variable 'y', the encoding of categorical labels is done alphabetically (Benign = 0, Malignant = 1)
    y_encoded = label_encoder.fit_transform(y)

    # transform tabular dataset to embeddings metrix
    table_to_embedding = TableToEmbedding(expended_col_names_mapper=COLUMN_NAME_TO_EXPANDED_NAME_MAPPER if use_expended_col_names else None,
                                          prefix=None, # in this case we pass False so we can manualy encode the predix separately 
                                          use_textual_numbers=use_textual_numbers)
    
    X_embeddings = table_to_embedding.transform(X)

    # enocode only the Prefix 
    prefix_embedding = table_to_embedding.sentence_transformer.encode(RECORD_EXTRA_INFO_PREFIX)

    # for each emcedding, concatenate the embedded prefix
    X_embeddings_with_prefix = [prefix_embedding + embdd for embdd in X_embeddings]

    X_train, X_test, y_train, y_test = train_test_split(X_embeddings_with_prefix, y_encoded, test_size=0.2, random_state=42)

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

    print("The prefix and the samples are encoded separatly and concatenated")
    print(
        f"###### with {'' if use_textual_numbers  else 'out'} textual numbers (1->'one') and"
        f" with {'' if use_expended_col_names else 'out'} expanded column names ######"
    )    
    print(classification_report(y_test, y_pred))


def run_brca_full_flow(
    add_informative_prefix: bool = False,
    use_expanded: bool = False,
):
    X,y = load_and_split_data("brca\\brca.csv")
    X.drop(labels= "Unnamed: 0",axis=1,inplace=True)
    
    label_encoder = LabelEncoder()
    # Encoding the target variable 'y', the encoding of categorical labels is done alphabetically
    y_encoded = label_encoder.fit_transform(y)

    # transform tabular dataset to embeddings metrix and the target to binary (Benign = 0, Malignant = 1)
    table_to_embedding = TableToEmbedding(
        add_informative_prefix,
        use_expanded,
    )
    X_embeddings = table_to_embedding.fit_transform(X)

    
    X_train,X_test,y_train,y_test= train_test_split(X_embeddings,y_encoded,test_size=0.2,random_state=42,)

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
        f"###### with {'' if add_informative_prefix  else 'out'} informative prefix and"
        f" with {'' if use_expanded else 'out'} expanded column names ######"
    )
    print(classification_report(y_test, y_pred))

def load_and_split_data(file_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(file_path)
    y = df[target_column]
    X = df.drop(
        labels=[target_column],
        axis=1,
    )
    return X,y
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
import pandas as pd
from sentence_transformers import (
    SentenceTransformer,
)
import inflect

MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

def get_textual_sample_by_params(
    expended_col_names_mapper:dict,
    prefix: str,
    text_nums: bool,
    col_name: str,
    row: pd.Series,
    num_2_word: inflect.engine,
):
    text_sample = f"{expended_col_names_mapper[col_name] if expended_col_names_mapper else col_name}:{num_2_word.number_to_words(row[col_name]) if text_nums else row[col_name]}"
    return (prefix + text_sample) if prefix else text_sample


class TableToText(
    BaseEstimator,
    TransformerMixin,
):
    """
    This transformer converts the dataset from table to a list of strings.
    It recives 3 boolean parameters, when set to true:

    add_informitive_prefix: concatenates an descriptive prefix before each record
    use_expended: uses an expended version of column name
    use_textual_numbers: transform each number to words (1->"one"). when set to false (1->"1")
    """

    def __init__(
        self,
        expended_col_names_mapper:dict,
        prefix: str,
        use_textual_numbers: bool,
    ):
        self.expended_col_names_mapper = expended_col_names_mapper
        self.prefix = prefix
        self.use_textual_numbers = use_textual_numbers
        self.textual_number_converter = inflect.engine()

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X: pd.DataFrame):
        X_copy: pd.DataFrame = X.copy()

        X_textual = [
            " ".join(
                [
                    get_textual_sample_by_params(
                        expended_col_names_mapper=self.expended_col_names_mapper,
                        prefix=self.prefix,
                        text_nums=self.use_textual_numbers,
                        col_name=col,
                        row=row,
                        num_2_word=self.textual_number_converter,
                    )
                    for col in X_copy.columns
                ]
            )
            for _, row in X_copy.iterrows()
        ]

        return X_textual


class TableToEmbedding(BaseEstimator,TransformerMixin):
    def __init__(
        self,
        expended_col_names_mapper:dict,
        prefix: str,
        use_textual_numbers: bool = False
    ):
        self.sentence_transformer = SentenceTransformer(MODEL)
        self.table_to_text = TableToText(
            expended_col_names_mapper=expended_col_names_mapper,
            prefix=prefix,
            use_textual_numbers=use_textual_numbers
        )

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame) -> list:
        X_copy: pd.DataFrame = X.copy()

        textual_records_list = self.table_to_text.transform(X_copy)
        embeddings = self.sentence_transformer.encode(textual_records_list)

        return embeddings

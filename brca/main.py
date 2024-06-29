import itertools
from  utils import run_separated_encoding_for_prefix_flow, run_brca_full_flow

if __name__ == "__main__":
    for state in list(
        itertools.product(
            [True, False],
            repeat=2,
        )
    ):
        run_brca_full_flow(*state)

    for state in list(
        itertools.product(
            [True, False],
            repeat=2,
        )
    ):
        # checks the separated encoding for prefix and samples with 2 params options:
        # 1. textual representation method (numeric string or textual)
        # 2. expended column name or original
        run_separated_encoding_for_prefix_flow(*state)
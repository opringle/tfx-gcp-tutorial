import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
import logging


def split_df(df: pd.DataFrame) -> List[pd.DataFrame]:
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    logging.info(
        "{} train, {} validation & {} test examples".format(
            len(train), len(val), len(test))
    )
    return [train, val, test]

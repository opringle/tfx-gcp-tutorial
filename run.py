from config import EnvironmentConfiguration


import pandas as pd

def load_data():
    df =pd.read_csv(
        './data/ramen-ratings.csv'
    )
    print(df.head())


if __name__ == '__main__':
    env_config = EnvironmentConfiguration()
    load_data()


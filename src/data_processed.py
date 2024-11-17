import pandas as pd
import yaml

with open("params.yml") as f:
    params = yaml.safe_load(f)

def load_data(data_path):
    return pd.read_csv(data_path)


def preprocess(data):
    ##preprocessing steps
    return data

if __name__ =="__main__":
    data = load_data(params["data_path"])
    processed_data = preprocess(data)
    processed_data.to_csv(params["processed_data_path"], index=False)
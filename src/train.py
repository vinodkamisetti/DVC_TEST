import pandas as pd
import yaml
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

with open("params.yml") as f:
    params = yaml.safe_load(f)

data = pd.read_csv(params["processed_data_path"])
X = data.drop("price", axis=1)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= params["train"]["test_size"], random_state=params["train"]["random_state"])

model = RandomForestRegressor(
    n_estimators=params["model_params"]["n_estimators"],
    max_depth=params["model_params"]["max_depth"],
    random_state=params["model_params"]["random_state"]
)

model.fit(X_train,y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
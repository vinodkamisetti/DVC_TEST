import pandas as pd
import yaml
import pickle
import json
from sklearn.metrics import mean_absolute_percentage_error, r2_score

with open("params.yml") as f:
    params = yaml.safe_load(f)

data = pd.read_csv(params["processed_data_path"])
X = data.drop("price" , axis=1)
y = data["price"]

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

prediction = model.predict(X)
map = mean_absolute_percentage_error(y, prediction)
r2 = r2_score(y,prediction)

metrics = {
    "map" : map,
    "r2" : r2 
}

with open("scores.json", "w") as f:
    json.dump(metrics,f)
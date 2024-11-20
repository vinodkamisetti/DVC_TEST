from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

#Load a model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(fource=True)

    try:
        bedroom = data["bedroom"],
        bathroom = data["bathroom"],
        sqft_living = data["sqft_living"],
        floor = data["floor"],
        yr_built = data["yr_built"]
        
        features = np.arry([bedroom,bathroom,sqft_living,floor,yr_built])

        pred = model.predict(features)
        prediction = np.round(pred)
        return jsonify({'predicted price': prediction[0]})
    except KeyError as e:
        return jsonify({'error': f'Missing features: {e.args[0]}'})
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__' :
    app.run(host='0.0.0.0', port=5000)
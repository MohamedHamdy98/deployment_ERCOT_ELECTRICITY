import pickle
from flask import Flask, request, jsonify
import pandas as pd 

# Reading  AI model
with open('ERCOT_Electricity_model.pkl', 'rb') as model_file:
    model, scaler_loading = pickle.load(model_file)

app = Flask(__name__)

@app.route('/predict', methods=['post'])
def predictt():

    data = request.json

    df = pd.DataFrame(data)
    
    df_scale = scaler_loading.transform(df)

    predictions = model.predict(df_scale)

    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

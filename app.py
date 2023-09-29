from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import torch
import pandas as pd
import numpy as np
from model import ShallowNet

app = Flask(__name__)
#allow all cross origin requests
CORS(app)
model = ShallowNet(4, 5, 1)
model.load_state_dict(torch.load('model.pth'))
model.eval()

"""
Endpoint for making predictions
Takes user input of number of days to predict Receipt_Count
Returns JSON object with date and predicted value
"""
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        num_days = int(data['days']) + 3

        #Preprocessing
        start_date = datetime(2021, 12, 29)
        date_list = [start_date + timedelta(days=x) for x in range(num_days)]
        df = pd.DataFrame(date_list, columns=['Date'])
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df['day_number'] = np.arange(len(df))
        df['day'] = df['Date'].str[8:].astype('int32')

        #read in data and get last three rows to build lag values
        temp_df = pd.read_csv("data_daily.csv")
        last_two_values = temp_df['Receipt_Count'].iloc[-2:].values.tolist()

        initial_lag_1 = last_two_values[1]
        initial_lag_2 = last_two_values[0]
        predictions = []
        
        #iterate over the number of days requested making prediction on each row
        for i in range(num_days):
            if i == 0:
                df.loc[i, 'lag_1'] = initial_lag_1
                df.loc[i, 'lag_2'] = initial_lag_2
            elif i == 1:
                df.loc[i, 'lag_1'] = predictions[-1]
                df.loc[i, 'lag_2'] = initial_lag_1 
            else:
                df.loc[i, 'lag_1'] = predictions[-1]
                df.loc[i, 'lag_2'] = predictions[-2]

            #convert row to tensor
            X_test_tensor = torch.tensor(df.loc[i, ['day_number', 'day', 'lag_1', 'lag_2']].values.astype('float32').reshape(1, -1), dtype=torch.float32)

            #make prediction based on gradient value at value
            with torch.no_grad():
                prediction = model(X_test_tensor).item()
                predictions.append(prediction)
        
        df["predictions"] = predictions
        df = df.dropna()
        return df.iloc[3:][["Date", "predictions"]].to_json(date_format='iso', orient='split')
        
    except Exception as e:
        return jsonify({"error": str(e)})

#run app
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port = 5001)

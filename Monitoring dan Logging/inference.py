import sys, os
import pandas as pd
import joblib
import numpy as np
from flask import Flask, request, render_template, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.automate_Sinta import SklearnPreprocessor

app = Flask(__name__)

model_path = 'C:/Users/Sinta/Documents/Larskar AI/Submission/Membangun Machine Learning/Eksperimen_SML_Sinta-Siti-Nuriah/SMSML_Sinta/model/linear_model.pkl'
model = joblib.load(model_path)

@app.route('/metrics')
def metrics_route():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Item_Weight': float(request.form['Item_Weight']),
        'Item_Visibility': float(request.form['Item_Visibility']),
        'Item_MRP': float(request.form['Item_MRP']),
        'Outlet_Establishment_Year': int(request.form['Outlet_Establishment_Year']),
        'Item_Fat_Content': request.form['Item_Fat_Content'],
        'Outlet_Size': request.form['Outlet_Size'],
        'Item_Type': request.form['Item_Type'],
        'Outlet_Location_Type': request.form['Outlet_Location_Type'],
        'Outlet_Type': request.form['Outlet_Type'],
        'Outlet_Identifier': request.form['Outlet_Identifier']
    }
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    pred = np.exp(prediction)
    return render_template('index.html', prediction_text=f"{pred:.2f} Sales")

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files.get('file')
    if not file:
        return "❌ Tidak ada file yang diupload.", 400

    try:
        # Baca CSV sebagai DataFrame
        df = pd.read_csv(file)

        # Prediksi log_sales
        pred_log_sales = model.predict(df)

        df['Predicted_Sales'] = np.exp(pred_log_sales)

        result_table = df.to_html(classes='table table-bordered', index=False)

        return render_template('index.html', tables=result_table)
    except Exception as e:
        return f"⚠️ Terjadi kesalahan saat memproses file: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

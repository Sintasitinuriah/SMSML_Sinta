from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.automate_Sinta import SklearnPreprocessor


app = Flask(__name__)

# Load pipeline model
model_path = 'C:/Users/Sinta/Documents/Larskar AI/Submission/Membangun Machine Learning/Eksperimen_SML_Sinta-Siti-Nuriah/Workflow-CI/models/linear_model.pkl'
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

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

        # Balik ke skala asli
        df['Predicted_Sales'] = np.exp(pred_log_sales)

        # Tampilkan sebagai HTML table
        result_table = df.to_html(classes='table table-bordered', index=False)

        return render_template('index.html', tables=result_table)
    except Exception as e:
        return f"⚠️ Terjadi kesalahan saat memproses file: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)

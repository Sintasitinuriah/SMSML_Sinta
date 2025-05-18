import sys, os, time, psutil, logging, threading
import pandas as pd
import joblib
import numpy as np

from flask import Flask, request, render_template
from prometheus_client import Counter, Gauge, Summary, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.automate_Sinta import SklearnPreprocessor

# --- Setup Logging ---
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

model_path = 'C:/Users/Sinta/Documents/Larskar AI/Submission/Membangun Machine Learning/Eksperimen_SML_Sinta-Siti-Nuriah/SMSML_Sinta/model/linear_model.pkl'
model = joblib.load(model_path)

# --- Prometheus Metrics ---
REQUESTS_TOTAL = Counter('requests_total', 'Total request masuk', ['endpoint', 'method'])
REQUESTS_FAILED = Counter('requests_failed_total', 'Total request gagal', ['endpoint'])

PREDICTION_TOTAL = Counter('prediction_total', 'Total prediksi berhasil')
PREDICTION_ERRORS = Counter('prediction_errors_total', 'Total prediksi error')
PREDICTION_PER_OUTLET = Counter('prediction_per_outlet', 'Prediksi per outlet', ['outlet_id'])

INFERENCE_LATENCY = Summary('inference_latency_seconds', 'Latency inferensi')

INPUT_DATA_SIZE = Gauge('input_data_size_bytes', 'Ukuran input CSV')
CSV_ROW_COUNT = Gauge('csv_row_count', 'Jumlah baris dalam CSV')
MODEL_LOAD_TIMESTAMP = Gauge('model_load_timestamp_seconds', 'Waktu model diload')

CPU_USAGE = Gauge('cpu_percent_usage', 'CPU usage %')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
FLASK_UPTIME = Gauge('flask_uptime_seconds', 'Flask uptime seconds')
ACTIVE_THREADS = Gauge('active_threads_total', 'Jumlah thread aktif')

start_time = time.time()
MODEL_LOAD_TIMESTAMP.set_to_current_time()

# --- Background Thread for Resource Monitoring ---
def monitor_resources():
    while True:
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used)
        ACTIVE_THREADS.set(threading.active_count())
        FLASK_UPTIME.set(time.time() - start_time)
        time.sleep(5)

threading.Thread(target=monitor_resources, daemon=True).start()

# --- Middleware untuk /metrics ---
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

@app.before_request
def before_request():
    REQUESTS_TOTAL.labels(request.endpoint, request.method).inc()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
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

        start = time.time()
        prediction = model.predict(input_df)[0]
        duration = time.time() - start

        INFERENCE_LATENCY.observe(duration)
        PREDICTION_TOTAL.inc()
        PREDICTION_PER_OUTLET.labels(outlet_id=data['Outlet_Identifier']).inc()

        pred = np.exp(prediction)
        logging.debug(f"Prediksi sukses. Latency: {duration:.4f}s")
        return render_template('index.html', prediction_text=f"{pred:.2f} Sales")

    except Exception as e:
        PREDICTION_ERRORS.inc()
        REQUESTS_FAILED.labels('/predict').inc()
        logging.error(f"Error prediksi: {e}")
        return render_template('index.html', prediction_text=f"Error during prediction: {e}")

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files.get('file')
    if not file:
        PREDICTION_ERRORS.inc()
        REQUESTS_FAILED.labels('/predict_csv').inc()
        return "❌ Tidak ada file yang diupload.", 400

    try:
        content = file.read()
        INPUT_DATA_SIZE.set(len(content))
        file.seek(0)

        df = pd.read_csv(file)
        CSV_ROW_COUNT.set(len(df))

        start = time.time()
        pred_log_sales = model.predict(df)
        duration = time.time() - start

        INFERENCE_LATENCY.observe(duration)
        PREDICTION_TOTAL.inc(len(df))

        for outlet in df['Outlet_Identifier'].unique():
            PREDICTION_PER_OUTLET.labels(outlet_id=outlet).inc()

        df['Predicted_Sales'] = np.exp(pred_log_sales)
        result_table = df.to_html(classes='table table-bordered', index=False)
        return render_template('index.html', tables=result_table)
    except Exception as e:
        PREDICTION_ERRORS.inc()
        REQUESTS_FAILED.labels('/predict_csv').inc()
        logging.error(f"Error saat prediksi CSV: {e}")
        return f"⚠️ Terjadi kesalahan saat memproses file: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

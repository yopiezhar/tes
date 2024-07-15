from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Muat model dan scaler
with open('model/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_heart_disease(model, scaler, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    # Masukkan data ke dalam numpy array
    user_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
    
    # Standarisasi data
    user_data = scaler.transform(user_data)
    
    # Lakukan prediksi
    prediction = model.predict(user_data)
    
    # Konversi hasil prediksi ke dalam bentuk yang dapat dibaca
    if prediction[0] == 1:
        result = "Risiko penyakit jantung"
    else:
        result = "Tidak ada risiko penyakit jantung"
    
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])
        
        result = predict_heart_disease(knn_model, scaler, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
        
        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=str(e))

if __name__ == "__main__":
    app.run(host='0.0.0.0')

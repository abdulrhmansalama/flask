from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# إنشاء مجلد uploads إذا لم يكن موجودًا
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# تنزيل نموذج ورم الدماغ من Google Drive
brain_model_url = 'https://drive.google.com/uc?id=1SpEOef-r-r_fG_np7nVQjP1zPhT1xMBz'
brain_model_path = 'brain_tumor_model.h5'

if not os.path.exists(brain_model_path):
    gdown.download(brain_model_url, brain_model_path, quiet=False)

# تحميل نموذج ورم الدماغ
brain_model = load_model(brain_model_path)
brain_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# دالة للتنبؤ بورم الدماغ
def predict_brain_tumor(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = brain_model.predict(img_array)
    predicted_class = brain_classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        predicted_class, confidence = predict_brain_tumor(file_path)
        
        return render_template('result.html', predicted_class=predicted_class, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)

import os
import cv2
import pickle
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from skimage.feature import hog

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

def extract_hog_features(image):
    # Calculate HOG features
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), visualize=False)
    return hog_features

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image
            img = cv2.imread(file_path)
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.Laplacian(img, cv2.CV_64F)
            img = extract_hog_features(img)
            img = img.reshape(-1, 1)
            img = img.reshape(1, -1)

            # Load the trained model
            ex3_model = pickle.load(open('rf_model_exp3.sav', 'rb'))

            # Make predictions
            exp3_result = ex3_model.predict(img)
            classes_names = ["Building", "Forest", "Glacier", "Mountains", "Sea", "Streets"]
            predicted_class = classes_names[exp3_result[0]]

            # Render the result template with the predicted class
            return render_template('result.html', predicted_class=predicted_class)

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
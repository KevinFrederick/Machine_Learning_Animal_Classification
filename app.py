from flask import Flask, render_template, request
from model import preprocess, predict
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def Home():
    return render_template('form.jinja')

@app.route('/', methods=['POST'])
def submit():
    image = request.files.get('images')
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)  # Secure the filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(file_path)  # Save the image file to the static/images folder

        res = predict(preprocess(image), file_path) 
    
        return render_template('result.jinja', result=res)
    return "Invalid file type", 400

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
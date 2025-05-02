from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from utils import validate_image
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'app/static/uploads'

#to create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {}
    if request.method == 'POST':
        # to check if the file is not present OR empty
        if 'photo' not in request.files or request.files['photo'].filename == '':
            result = {'Error': 'No file uploaded'}
        else:
            file = request.files['photo'] #retrieving the file
            filename = secure_filename(file.filename) #to secure filename
            filepath = os.path.join(UPLOAD_FOLDER, filename) #path to save the file
            file.save(filepath)
            result = validate_image(filepath) #validating the image
    return render_template('index.html', result=result) 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
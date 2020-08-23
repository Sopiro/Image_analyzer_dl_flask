from flask import Flask, render_template, request, url_for, flash, redirect
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.debug = True
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.secret_key = 'sopiro'


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/fileUpload', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        f = request.files['file']

        if f.filename == '':
            return redirect(request.url)

        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return render_template('result.html', filename=filename)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(host='0.0.0.0')

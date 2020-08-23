from flask import Flask, render_template, request, url_for, flash, redirect
import os
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
app.debug = True
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.secret_key = 'sopiro'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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

        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return render_template('result.html', filename=filename)
        else:
            return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/delete/<filename>')
def delete_file(filename):
    try:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    except FileNotFoundError:
        pass
    return redirect(url_for('index'))

# @app.teardown_request
# def teardown_request(ex):
#     for f in os.listdir(app.config['UPLOAD_FOLDER']):
#         os.remove(f)


if __name__ == '__main__':
    app.run(host='0.0.0.0')

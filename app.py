from flask import Flask, render_template, request, url_for, flash, redirect, jsonify
import os
from werkzeug.utils import secure_filename
import requests
import json

from caption_generator import generate_caption
from ocr import kakao_ocr
from color_analyzer import analyze_color
from translator import kakao_translator

app = Flask(__name__)
app.debug = False
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.secret_key = 'sopiro'

do_translation = False
remain_upload_image = True
KAKAO_API_KEY = '1b9ef11c3bdeaa8cb71013c0e2ecb9f9'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

ai_program = {'color': analyze_color,
              'ocr': kakao_ocr,
              'caption': generate_caption}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/rest/<mode>', methods=['POST'])
def rest(mode):
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file', 500

        f = request.files['file']

        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.abspath('static/uploads/' + filename)

            if mode == 'color':
                result = analyze_color(image_path)
            elif mode == 'ocr':
                result = kakao_ocr(image_path, KAKAO_API_KEY)
            elif mode == 'caption':
                result = generate_caption(image_path)

                if do_translation:
                    result = kakao_translator('it seems like ' + result, KAKAO_API_KEY)
            else:
                return 'Mode error: ' + mode, 500
            print(result)

            if not remain_upload_image:
                os.remove(image_path)

            return json.dumps({'result': result}, ensure_ascii=False), 200
        else:
            return 'Not allowed file format', 500


@app.route('/fileUpload', methods=['POST'])
def file_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect('/')

        f = request.files['file']

        if f.filename == '':
            flash('파일을 선택해 주세요')
            return redirect('/')

        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            image_path = os.path.abspath('static/uploads/' + filename)
            caption = generate_caption(image_path)
            pre = 'it seems like '

            if do_translation:
                korean_caption = kakao_translator(pre + caption)

                return render_template('result.html', filename=filename, en_caption=caption, kr_caption=korean_caption)
            else:
                return render_template('result.html', filename=filename, en_caption=caption)
        else:
            return redirect('/')


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
    if not os.path.exists("static"):
        os.mkdir('static')
        os.mkdir('static/uploads')

    # print(os.path.abspath('static/uploads/image.jpg'))
    # print(generate_caption(os.path.abspath('static/uploads/image.jpg')))
    # print(generate_caption(os.path.abspath('static/uploads/elephant.jpg')))

    app.run(host='0.0.0.0')

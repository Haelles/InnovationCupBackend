import io
import os
from flask import Flask, make_response, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import json
from imageio import imread
import base64

UPLOAD_FOLDER = '../temp/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

name_list = ['original.jpg', 'sketch.png', 'mask.png', 'stroke.png']
names = ['original', 'sketch', 'mask', 'stroke']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        filename = request.form['filename']
        with open(UPLOAD_FOLDER + filename, 'rb') as original_image:
            # print(type(original_image))  # <class '_io.BufferedReader'>
            image64 = base64.b64encode(original_image.read())  # 二进制串
            with open(UPLOAD_FOLDER + 'test_original.jpg', 'wb') as decode_image:
                decode_image.write(base64.b64decode(image64))
                return 'true'

    return 'false'


@app.route('/result/<path:file_name>', methods=['GET', 'POST'])
def index(file_name):
    if os.path.isdir(file_name):
        return '<h1>文件夹无法下载</h1>'
    else:
        return send_from_directory(UPLOAD_FOLDER, file_name, as_attachment=True)


def get_base64():
    for i in range(4):
        with open(UPLOAD_FOLDER + name_list[i], 'rb') as original_image:
            # print(type(original_image))  # <class '_io.BufferedReader'>
            image64 = base64.b64encode(original_image.read())  # 二进制串
            with open(UPLOAD_FOLDER + names[i], 'wb') as res:
                res.write(image64)


if __name__ == '__main__':
    # app.run()
    get_base64()

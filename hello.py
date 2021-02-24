import os
from flask import Flask, make_response, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import json

UPLOAD_FOLDER = './api'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # file = cv2.imread('./uploads/' + filename)
            #             resp = make_response(file)
            data = {'result': filename}
            return json.dumps(data)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/result/<path:file_name>', methods=['GET', 'POST'])
def index(file_name):
    if os.path.isdir(file_name):
        return '<h1>文件夹无法下载</h1>'
    else:
        return send_from_directory(UPLOAD_FOLDER, file_name, as_attachment=True)


if __name__ == '__main__':
    app.run()

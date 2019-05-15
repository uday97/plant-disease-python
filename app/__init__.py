import os
from os.path import dirname as up
from flask import Flask, render_template, url_for, request, redirect
# from werkzeug import secure_filename

app = Flask(__name__, static_url_path='/uploads')

UPLOAD_FOLDER = os.getcwd()+'/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/file_upload', methods=['GET', 'POST'])
def file_upload():
    file = request.files['file']
    # filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg'))
    from Start import main
    output = main()
    return render_template('result.html', data = output)

if __name__ == '__main__':
    app.run(debug=True)
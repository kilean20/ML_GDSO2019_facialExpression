# import csv
# import zipfile
# import pandas as pd
# import numpy as np
# import seaborn as sns 


from video_cam import VideoCamera
from keras.models import load_model
from flask import Flask, render_template, Response

%tb
face_model = load_model('data/VGGfaceTransfer.warmUp.cosDecay.LabelSmoothing.hdf5')

app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')

def gen(camera):
	while True:
		frame = camera.get_frame(face_model)
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
	return Response(gen(VideoCamera()),
					mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
	app.run(host='localhost',port = 2000, debug=True)
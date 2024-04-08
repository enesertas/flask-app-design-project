from flask import Flask, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import cv2
import time
import sys
import os

app = Flask(__name__)
CORS(app)
url_public = sys.argv[1]

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html', url_public = url_public)

@app.route("/generate-image", methods=['GET'])
def generate():
    
    # generate syncdiffusion image, encode in base64 and send in json
    os.system("""python3 ./flask-app-design-project/SyncDiffusion/sample_syncdiffusion.py \
--prompt "a photo of a mountain range at twilight" \
--negative "" \
--H 512 \
--W 3072 \
--seed 100 \
--steps 50 \
--sync_weight 20.0 \
--sync_decay_rate 0.95 \
--sync_freq 1 \
--sync_thres 10 \
--sd_version "2.0" \
--save_dir "results" \
--stride 16""")
    return send_file("./templates/lpips_A_cinematic_view_of_a_castle_in_the_sunset.jpg", as_attachment="lpips_A_cinematic_view_of_a_castle_in_the_sunset.jpg", mimetype="image/jpg")

if __name__ == "__main__":
    app.run()
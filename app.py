from flask import Flask, jsonify, send_file, render_template
from flask_cors import CORS, cross_origin
from PIL import Image
from io import BytesIO
import base64
import cv2
import time

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/generate-image", methods=['GET'])
def generate():
    
    # generate syncdiffusion image, encode in base64 and send in json
    time.sleep(3)
    return send_file("./templates/lpips_A_cinematic_view_of_a_castle_in_the_sunset.jpg", as_attachment="lpips_A_cinematic_view_of_a_castle_in_the_sunset.jpg", mimetype="image/jpg")

if __name__ == "__main__":
    app.run()
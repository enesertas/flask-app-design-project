from flask import Flask, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import cv2
import time
import sys
import os
import torch

from syncdiffusion.syncdiffusion_model import SyncDiffusion
from syncdiffusion.utils import seed_everything

app = Flask(__name__)
CORS(app)
url_public = sys.argv[1]
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Load SyncDiffusion model
syncdiffusion_model = SyncDiffusion(device, sd_version="2.0")

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html', url_public = url_public)

@app.route("/generate-image", methods=['GET'])
def generate():

    seed_everything(100)

    # Generate images
    img = syncdiffusion_model.sample_syncdiffusion(
        prompts = "a photo of a mountain range at twilight",
        negative_prompts = "",
        height = 512,
        width = 3072,
        num_inference_steps = 20,
        guidance_scale = 7.5,
        sync_weight = 20.0,
        sync_decay_rate = 0.95,
        sync_freq = 1,
        sync_thres = 10,
        stride = 16,
    )
    img.save("./flask-app-design-project/templates/result.jpg")
    print(f"[INFO] saved the result")
    
    return send_file("/templates/result.jpg", as_attachment="result.jpg", mimetype="image/jpg")

if __name__ == "__main__":
    app.run()
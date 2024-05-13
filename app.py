from flask import Flask, jsonify, send_file, render_template, request
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

@app.route("/generate-image", methods=['GET', 'POST'])
def generate():

    prompt = request.args.get('prompt')
    seed_everything(100)
    syncdiffusion = SyncDiffusion(device, sd_version="2.0")
    # Hedi
    img = syncdiffusion.sample_syncdiffusion(
        prompts = prompt,
        negative_prompts = "",
        height = 128,
        width = 256,  # 1024, 3072
        #latent_size = 64, # for a 512x512 model (96 for a 768x768 model)
        num_inference_steps = 10,
        guidance_scale = 7.5,
        sync_weight = 20.0,
        sync_decay_rate = 0.95,
        # sync_thres = 10,
        # sync_freq = 1,
        # stride = 16,
        WFov=150,
        HFov=100,
        h_patch=32,
        w_patch=32,
        n_rows=3,
        # loop_closure = False
    )
    
    # # Generate images
    # img = syncdiffusion_model.sample_syncdiffusion(
    #     prompts = prompt,
    #     negative_prompts = "",
    #     height = 512,
    #     width = 3072,
    #     num_inference_steps = 20,
    #     guidance_scale = 7.5,
    #     sync_weight = 20.0,
    #     sync_decay_rate = 0.95,
    #     sync_freq = 1,
    #     sync_thres = 10,
    #     stride = 16,
    # )
    img.save("./templates/result.jpg")
    print(f"[INFO] saved the result")
    return send_file("./templates/result.jpg", as_attachment="result.jpg", mimetype="image/jpg")

if __name__ == "__main__":
    app.run()
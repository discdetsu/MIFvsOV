import argparse
import io
import os
from PIL import Image
import cv2
import numpy

import torch
from flask import Flask, render_template, request, redirect, send_file

app = Flask(__name__)

model = torch.hub.load('yolov5-master', 
                                        'custom',
                                        source='local', 
                                        path='best.pt', 
                                        force_reload=True) # force_reload = recache latest code
model.eval()

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        img = Image.open(request.files['file'].stream)
        results = model(img, size=640)

        # for debugging
        # data = results.pandas().xyxy[0].to_json(orient="records")
        # return data

        results.render()  # updates results.imgs with boxes and labels

        res = cv2.cvtColor(results.imgs[0], cv2.COLOR_BGR2RGB)
        img_encode = cv2.imencode('.jpg', res)[1].tobytes()

        return send_file(io.BytesIO(img_encode), download_name='image.jpg',mimetype='image/jpg')

    return render_template("index.html")


if __name__ == "__main__":


    app.run(debug=True)  # debug=True causes Restarting with stat

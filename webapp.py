import argparse
import io
import os
from PIL import Image
import cv2
import numpy

import torch
from flask import Flask, render_template, request, redirect, send_file

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
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
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    # parser.add_argument("--port", default=8080, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 
                                            'custom', 
                                            path='best.pt', 
                                            force_reload=True) # force_reload = recache latest code
    model.eval()
    app.run(debug=True)  # debug=True causes Restarting with stat

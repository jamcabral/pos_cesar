import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "Jammesson/rnn"
FILENAME = "best.pt"

yolov5_weights = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=yolov5_weights, force_reload=True)


def yolo(im, size=640):
    # g = (size / max(im.size))  # gain
    # im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)  # resize
    results = model(im)  # inference
    results.render()  # updates results.imgs with boxes and labels
    return Image.fromarray(results.ims[0])


title = "Carro, moto ou caminhão detector"
description = """Esse modelo é para detectar o que é carro, moto ou caminhão.
"""

inputs = gr.inputs.Image(shape=(640, 640), type='pil', label="Original Image")
outputs = gr.outputs.Image(type="pil", label="Output Image")

gr.Interface(
    fn=yolo,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    examples=[["cam1.jpeg"], ["cam2.jpeg"], ["carro1.jpeg"], ["carro2.jpeg"],
              ["moto1.jpeg"], ["moto2.jpeg"]]
).launch(debug=True)

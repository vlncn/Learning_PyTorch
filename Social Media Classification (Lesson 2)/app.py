import gradio as gr
from fastai.vision.all import *
import pathlib
plt = platform.system()
if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

gr.Interface(fn=predict, inputs=gr.Image(height=512, width=512), outputs=gr.Label(num_top_classes=4)).launch(share=True)
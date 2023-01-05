import gradio as gr
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/historyAbisko.png", overwrite=True)
dataset_api.download("Resources/images/historySpånga.png", overwrite=True)
dataset_api.download("Resources/images/historyUppsala.png", overwrite=True)

dataset_api.download("Resources/images/predictionsAbisko.png", overwrite=True)
dataset_api.download("Resources/images/predictionsSpånga.png", overwrite=True)
dataset_api.download("Resources/images/predictionsUppsala.png", overwrite=True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Textbox("Historical Predictions for Abisko", label="")
            input_img = gr.Image("historyAbisko.png", elem_id="h-img1")
        with gr.Column():
            gr.Textbox("Historical Predictions for Spånga", label="")
            input_img = gr.Image("historySpånga.png", elem_id="h-img2")
        with gr.Column():
            gr.Textbox("Historical Predictions for Uppsala", label="")
            input_img = gr.Image("historyUppsala.png", elem_id="h-img3")
    with gr.Row():
        with gr.Column():
            gr.Textbox("Future Predictions for Abisko", label="")
            input_img = gr.Image("predictionsAbisko.png", elem_id="p-img1")
        with gr.Column():
            gr.Textbox("Future Predictions for Spånga", label="")
            input_img = gr.Image("predictionsSpånga.png", elem_id="p-img2")
        with gr.Column():
            gr.Textbox("Future Predictions for Uppsala", label="")
            input_img = gr.Image("predictionsUppsala.png", elem_id="p-img3")

demo.launch()
import gradio as gr

def predict_image(img):
    return "Your model prediction here"

gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(lines=10)
).launch()

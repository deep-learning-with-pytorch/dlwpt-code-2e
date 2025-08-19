import gradio as gr

def hello_world(name):
    return "Hello, " + name + "!"

demo = gr.Interface(
    fn=hello_world,
    inputs=["text"],
    outputs=["text"],
    flagging_mode="auto",
)

demo.launch()
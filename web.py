from flask import Flask, request, render_template
from diffusers import StableDiffusionPipeline
import torch

app = Flask(__name__)

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to(device)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]
        output = pipe(prompt)
        print(output)  # Debugging line to check the output structure
        image = output["images"][0]  # Assuming 'images' is the correct key
        image_path = "static/generated_cartoon.png"
        image.save(image_path)
        return render_template("index.html", image_path=image_path)
    return render_template("index.html", image_path=None)

if __name__ == "__main__":
    app.run(debug=True)

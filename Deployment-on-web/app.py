#!/usr/bin/env python
#coding=utf-8


# Import modules
import os
import io
import torch
from urllib.request import urlopen
from PIL import Image
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, send_from_directory, jsonify

# Load a pre-trainied models from torchvision.models
models_dict = {
    'DenseNet': models.densenet121(pretrained=True),
    'ResNet': models.resnet18(pretrained=True),
    'VGG': models.vgg16(pretrained=True)
}

# Set models to evaluation mode
for model in models_dict.values():
    model.eval()

# Load the class labels from a file
class_labels_url = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)
class_labels = urlopen(class_labels_url).read().decode("utf-8").split("\n")

# Define the transformation of the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(model, transform, image, class_labels):
  # Transform the image and convert it to a tensor
  image_tensor = transform(image).unsqueeze(0)

  # Pass the image through the model
  with torch.no_grad():
      output = model(image_tensor)

  # Apply softmax to get probabilities
  probabilities = F.softmax(output, dim=1)

  # Select the class with the higherst probability
  class_id = torch.argmax(output).item()
  class_name = class_labels[class_id]

  # Get the probability of the predicted class
  class_probability = round(probabilities[0, class_id].item(),2)*100
  
  return class_name, class_probability


app = Flask(__name__)

@app.route("/")
def home():
  return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    selected_option = request.form.get('option')
    image_file = request.files.get('file')
    
    if not image_file:
        return jsonify({'error': 'No image file provided'}), 400

    if selected_option not in models_dict:
        return jsonify({'error': 'Invalid model selected'}), 400
    
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Select the model based on the selected option
    model = models_dict[selected_option]

    # Perform image prediction
    class_name, class_probability = predict(model, transform, image, class_labels)

    # Example processing based on the selected option
    result = f"I think this is: {class_name} with {class_probability}% probability."
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=os.getenv("PORT", 5000))
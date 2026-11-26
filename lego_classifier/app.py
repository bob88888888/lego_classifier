import torch
import torch.nn as nn
import gradio as gr
import pandas as pd

from torchvision import models
from PIL import Image

from finetuning_functions.TransformForPrediction import transform_image

device = "cpu"

class_data=pd.read_csv("dataset_mapping_1.csv")

class_names = dict(zip(class_data["class ID"], class_data["Class"]))

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    in_features=model.classifier[1].in_features,
    out_features=14
)

model.load_state_dict(
    torch.load(
        "models/EfficientnetB0_50_epoch_model.pth",
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

def predict(image):

    # Preprocess uploaded image
    image_tensor = transform_image(image)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Make prediction
    with torch.inference_mode():

        logits = model(image_tensor)

        probabilities = torch.softmax(logits, dim=1)[0]

    return {
        class_names[i]: probabilities[i].item()
        for i in range(len(probabilities))
    }


# Gradio interface
demo = gr.Interface(
    fn=predict,

    inputs=gr.Image(
        type="pil",
        label="Upload LEGO brick image"
    ),

    outputs=gr.Label(
        num_top_classes=3,
        label="Predicted LEGO brick"
    ),

    title="LEGO Brick Classifier",

    description="Identify 14 different types of LEGO bricks "
                "using a fine-tuned EfficientNet-B0 model."
)


if __name__ == "__main__":
    demo.launch()

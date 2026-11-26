import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from PIL import Image
from torchvision import models, transforms

def transform_image(image):

    image = image.convert("RGB")

    image_array = np.array(image)

    # Replace white background with black
    mask = np.all(image_array > 230, axis=2)
    image_array[mask] = [0, 0, 0]

    processed_image = Image.fromarray(image_array)

    # Crop empty space
    bbox = processed_image.getbbox()

    if bbox is not None:
        processed_image = processed_image.crop(bbox)

    # Shrink while preserving aspect ratio
    processed_image.thumbnail((150, 150))

    # Create a black canvas
    canvas = Image.new("RGB", (224, 224), (0, 0, 0))

    # Calculate centre position
    x = (224 - processed_image.width) // 2
    y = (224 - processed_image.height) // 2

    canvas.paste(processed_image, (x, y))

    # Define preprocessing
    predict_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = predict_transform(canvas)

    return image_tensor

def load_model(device="cpu"):

    loaded_model = models.efficientnet_b0(weights=None)

    loaded_model.classifier[1] = nn.Linear(
        loaded_model.classifier[1].in_features,
        14
    )

    loaded_model.load_state_dict(
        torch.load(
            "models/FineTuned_EffnetB0_full_dataset_10_epoch2.pth",
            map_location=device,
            weights_only=True
        )
    )

    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    return loaded_model
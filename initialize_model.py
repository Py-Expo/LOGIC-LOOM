"""
Script to initialize and save a ResNet50 model for Brain Tumor Classification
This creates an untrained model structure that can be loaded by the Flask app.
Note: For accurate predictions, you'll need to train this model on the brain tumor dataset.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

print("Initializing ResNet50 model...")

# Create ResNet50 model with pretrained weights
resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Make parameters trainable
for param in resnet_model.parameters():
    param.requires_grad = True

# Modify the final fully connected layer for 4 classes
n_inputs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Linear(n_inputs, 2048),
    nn.SELU(),
    nn.Dropout(p=0.4),
    nn.Linear(2048, 2048),
    nn.SELU(),
    nn.Dropout(p=0.4),
    nn.Linear(2048, 4),
    nn.LogSigmoid()
)

# Make all parameters trainable
for name, child in resnet_model.named_children():
    for name2, params in child.named_parameters():
        params.requires_grad = True

# Save the model
model_path = './models/bt_resnet50_model.pt'
torch.save(resnet_model.state_dict(), model_path)

print(f"✓ Model initialized and saved to {model_path}")
print("\nNote: This is a model with pretrained ResNet50 weights but untrained")
print("      classification head. For accurate brain tumor predictions, you need to:")
print("      1. Download the brain tumor dataset from: https://figshare.com/articles/brain_tumor_dataset/1512427")
print("      2. Run the notebook 'torch_brain_tumor_classifier.ipynb' to train the model")
print("      3. The trained model will give much better predictions")
print("\nHowever, the Flask app can now start and accept requests!")

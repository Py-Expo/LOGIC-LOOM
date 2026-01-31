from io import BytesIO 
from torch import argmax, load
from torch import device as DEVICE
from torch.cuda import is_available
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models import resnet50
from flask import Flask, jsonify, request, render_template, redirect, url_for
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning) 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
LABELS = ['None', 'Meningioma', 'Glioma', 'Pitutary']

device = "cuda" if is_available() else "cpu"

resnet_model = resnet50(weights='ResNet50_Weights.DEFAULT')

for param in resnet_model.parameters():
    param.requires_grad = True

n_inputs = resnet_model.fc.in_features
resnet_model.fc = Sequential(Linear(n_inputs, 2048),
                            SELU(),
                            Dropout(p=0.4),
                            Linear(2048, 2048),
                            SELU(),
                            Dropout(p=0.4),
                            Linear(2048, 4),
                            LogSigmoid())

for name, child in resnet_model.named_children():
    for name2, params in child.named_parameters():
        params.requires_grad = True

resnet_model.to(device)
resnet_model.load_state_dict(load('./models/bt_resnet50_model.pt', map_location=DEVICE(device)))
resnet_model.eval()

def preprocess_image(image_bytes):
  transform = Compose([Resize((512, 512)), ToTensor()])
  img = Image.open(BytesIO(image_bytes))
  return transform(img).unsqueeze(0)

def get_prediction(image_bytes):
  tensor = preprocess_image(image_bytes=image_bytes)
  y_hat = resnet_model(tensor.to(device))
  class_id = argmax(y_hat.data, dim=1)
  return str(int(class_id)), LABELS[int(class_id)]

@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    file = request.files.get('bt_image')
    if file:
      # Save the uploaded file
      filename = file.filename
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      file.save(filepath)
      
      # Read the file for prediction
      with open(filepath, 'rb') as f:
        img_bytes = f.read()
      
      class_id, class_name = get_prediction(img_bytes)
      return render_template('pred.html', pred=class_name, f_name=filename)
  return render_template('index.html')

@app.route('/empty_page')
def empty_page():
  return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    file = request.files['file']
    img_bytes = file.read()
    class_id, class_name = get_prediction(img_bytes)
    return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
  app.run()

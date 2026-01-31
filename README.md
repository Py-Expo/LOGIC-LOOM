# 🧠 Brain Tumor Analysis System

A sophisticated web-based application for brain tumor classification using deep learning. This project leverages transfer learning with ResNet50 to classify MRI brain images into four categories: No Tumor, Meningioma, Glioma, and Pituitary tumors.

## 📋 Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset Information](#dataset-information)
- [Important Notes](#important-notes)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## ✨ Features

- **Web-based Interface**: Modern, responsive UI with gradient animations and professional design
- **Real-time Image Classification**: Instant brain tumor classification from uploaded MRI scans
- **Transfer Learning**: Powered by ResNet50 architecture with pretrained ImageNet weights
- **Four-class Classification**: Distinguishes between:
  - No Tumor
  - Meningioma
  - Glioma
  - Pituitary tumors
- **User-Friendly Design**: Intuitive file upload with visual feedback and clear results display
- **Mobile Responsive**: Fully responsive design for desktop, tablet, and mobile devices
- **Professional UI/UX**: Modern gradient animations, smooth transitions, and polished visual elements

---

## 🛠 Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | Flask 3.1.2 |
| **Deep Learning** | PyTorch 2.10.0 |
| **Computer Vision** | torchvision 0.25.0 |
| **Image Processing** | Pillow 12.1.0 |
| **Numerical Computing** | NumPy 2.4.1 |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Python Version** | 3.8+ |

---

## 📁 Project Structure

```
Brain Tumor Analysis/
├── app.py                              # Flask application main file
├── initialize_model.py                 # Script to create untrained model
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── models/
│   └── bt_resnet50_model.pt           # Trained/untrained model weights
├── static/
│   ├── css/
│   │   └── main.css                   # Styled CSS with animations
│   ├── images/
│   │   ├── brain_logo.png             # Project logo
│   │   └── (uploaded images)          # User uploaded MRI scans
├── templates/
│   ├── index.html                     # Home page with upload form
│   └── pred.html                      # Results page with classification
├── test_images/                       # Sample test images
├── venv/                              # Python virtual environment
└── brain_tumor_dataset_preparation.ipynb  # Dataset preparation notebook
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 2GB RAM minimum
- CUDA 12.x (optional, for GPU acceleration)

### Step 1: Clone or Navigate to Project
```bash
cd "Brain Tumor Analysis"
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Initialize Model (First Time Only)
```bash
python initialize_model.py
```
This creates the untrained model architecture saved to `models/bt_resnet50_model.pt`.

---

## 💻 Usage

### Starting the Application

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate

# Run Flask development server
python app.py
```

The application will start on `http://127.0.0.1:5000`

### Using the Web Interface

1. **Open Browser**: Navigate to `http://127.0.0.1:5000`
2. **Upload Image**: Click the upload area or drag-and-drop a brain MRI scan (JPG, PNG, etc.)
3. **Analyze**: Click "Analyze Image" button
4. **View Results**: See classification result with color-coded output:
   - 🟢 **Green**: No Tumor
   - 🟠 **Orange**: Meningioma
   - 🔴 **Red**: Glioma
   - 🟣 **Purple**: Pituitary tumor
5. **Analyze Another**: Click button to return to home and analyze another image

---

## 🧠 Model Architecture

### ResNet50 Backbone
- **Base Model**: ResNet50 with ImageNet pretrained weights
- **Input Size**: 512×512 RGB images
- **Preprocessing**: 
  - Resize to 512×512
  - Normalize with ImageNet mean/std
  - Convert to tensor

### Classification Head
- **Layer 1**: 2048 → 512 (ReLU activation)
- **Dropout**: 0.5 (prevents overfitting)
- **Layer 2**: 512 → 256 (ReLU activation)
- **Dropout**: 0.3
- **Layer 3**: 256 → 128 (ReLU activation)
- **Output Layer**: 128 → 4 (softmax for multi-class classification)

### Model Summary
```
Total Parameters: ~25.5 Million
Trainable Parameters: ~3.3 Million (classification head)
Frozen Parameters: ~23.5 Million (ResNet50 backbone)
```

---

## 📊 Dataset Information

### Brain Tumor Dataset
The model is designed to work with the **Brain Tumor MRI Dataset** from Figshare:
- **Source**: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **Classes**: 4 types (No Tumor, Meningioma, Glioma, Pituitary)
- **Image Count**: ~7,000 total images
- **Image Format**: JPG/PNG
- **Image Dimensions**: Variable (preprocessing handles resizing)

### Training Recommendations
```
- Training-Validation Split: 70-30
- Batch Size: 32-64
- Learning Rate: 0.0001 (Adam optimizer)
- Epochs: 50-100
- Loss Function: CrossEntropyLoss
- Early Stopping: Monitor validation accuracy
```

---

## ⚠️ Important Notes

### Current Model Status
⚠️ **The model shipped with this project is NOT trained**
- The `initialize_model.py` script creates an untrained ResNet50 architecture
- Random predictions are returned until proper training is performed
- **For resume/production use**, you must train the model on actual brain tumor data

### Training the Model
To use this project effectively:

1. **Obtain Dataset**: Download from Kaggle or Figshare
2. **Prepare Data**: Use `brain_tumor_dataset_preparation.ipynb` notebook
3. **Train Model**: 
   - Use PyTorch training loop
   - Modify `app.py` to load trained weights
   - Test on validation set (aim for 90%+ accuracy)
4. **Evaluate**: Test with medical MRI samples
5. **Deploy**: Update model path in Flask app

### Disclaimer
This tool is **experimental** and should **NOT be used** for:
- Actual medical diagnosis
- Clinical decision making
- Patient treatment planning

Always consult qualified medical professionals for brain tumor diagnosis and treatment.

---

## 🔧 Configuration

### Modify Classification Labels
Edit in `app.py` line ~45:
```python
class_names = ['None', 'Meningioma', 'Glioma', 'Pituitary']
```

### Change Upload Directory
Edit in `app.py` line ~30:
```python
UPLOAD_FOLDER = 'static/images'
```

### Adjust Image Size
Edit in `app.py` line ~35:
```python
IMG_SIZE = 512  # Change as needed
```

---

## 📝 API Endpoints

### GET `/`
- **Description**: Render home page
- **Response**: HTML page with upload form

### POST `/`
- **Description**: Upload image and get classification
- **Parameters**: Form data with `bt_image` file
- **Response**: HTML results page with prediction

### POST `/predict`
- **Description**: API endpoint for JSON predictions
- **Request**: JSON with base64 encoded image
- **Response**: JSON with classification results

---

## 🎨 UI/UX Features

### Design Highlights
- **Color Scheme**: Professional navy blue (#0f172a) and medical teal (#0891b2)
- **Animations**: Smooth fade-in, slide-up, and pulse animations
- **Responsive Design**: Works on 320px to 4K+ screens
- **Accessibility**: High contrast ratios and readable typography
- **Interactive Feedback**: Hover effects on buttons and cards

### CSS Variables
Customize colors in `static/css/main.css`:
```css
--primary: #0f172a;           /* Navy blue */
--secondary: #0891b2;          /* Medical teal */
--accent-success: #059669;     /* Green */
--accent-warning: #d97706;     /* Orange */
--accent-danger: #dc2626;      /* Red */
```

---

## 🚀 Future Improvements

### Phase 1: Model Enhancement
- [ ] Train ResNet50 on brain tumor dataset
- [ ] Implement class imbalance handling
- [ ] Add confidence scores to predictions
- [ ] Create model evaluation metrics dashboard

### Phase 2: Feature Expansion
- [ ] Image preprocessing filters (brightness, contrast adjustment)
- [ ] Batch image upload and analysis
- [ ] Export results to PDF/CSV
- [ ] User authentication and history
- [ ] Dark mode toggle

### Phase 3: Advanced Features
- [ ] Explainable AI (Grad-CAM visualization)
- [ ] Ensemble models for better accuracy
- [ ] DICOM file support
- [ ] Real-time prediction API
- [ ] Docker containerization

### Phase 4: Production Ready
- [ ] Unit and integration testing
- [ ] Error handling and logging
- [ ] Load balancing for multiple users
- [ ] Database for predictions history
- [ ] API rate limiting
- [ ] Deployment to cloud (AWS/GCP/Azure)

---

## 📖 File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main Flask application with routes and model inference |
| `initialize_model.py` | Creates untrained model weights for development |
| `brain_tumor_dataset_preparation.ipynb` | Jupyter notebook for dataset preparation and exploration |
| `requirements.txt` | Python package dependencies |
| `templates/index.html` | Home page template with upload form |
| `templates/pred.html` | Results page template showing predictions |
| `static/css/main.css` | Comprehensive styling with animations and gradients |
| `static/images/` | Directory for uploaded and logo images |

---

## 🔍 Troubleshooting

### Issue: Model file not found
```
Solution: Run python initialize_model.py to create model weights
```

### Issue: Port 5000 already in use
```bash
# Kill process using port 5000
lsof -ti:5000 | xargs kill -9

# Or run on different port
python -c "from app import app; app.run(port=5001)"
```

### Issue: CUDA out of memory
```python
# Use CPU instead in app.py
device = torch.device('cpu')
```

### Issue: Image upload not working
```
Solution: Ensure templates/ folder exists and check Flask upload folder permissions
```

---

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Brain Tumor Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

## 📄 License

This project is provided as-is for educational and portfolio purposes. 

**Medical Disclaimer**: This tool is not approved for medical diagnosis. Always consult qualified medical professionals for any health-related decisions.

---

## 👤 Author

**Adithya**  
Portfolio Project: Brain Tumor Analysis System  

---

## 💬 Key Skills Demonstrated

- ✅ Deep Learning (PyTorch, Transfer Learning)
- ✅ Full-Stack Web Development (Flask, HTML/CSS/JS)
- ✅ Computer Vision (ResNet50, Image Classification)
- ✅ UI/UX Design (Responsive, Modern CSS)
- ✅ Project Structure & Documentation
- ✅ Version Control & Best Practices

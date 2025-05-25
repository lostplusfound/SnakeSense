# 🐍 SnakeSense - Snake Species Classifier

**SnakeSense** is an AI-powered snake species classifier designed to identify snake species and determine if a snake is venomous. This initial release includes a trained model exported in both FastAI (`.pkl`) and ONNX (`.onnx`) formats.

> ⚠️ **Disclaimer**: This model is a **work in progress** and is not production-ready. Use with caution, especially in real-world scenarios involving venomous snakes.

---

## 📦 Model Details

* **Architecture**: ResNet101
* **Framework**: FastAI
* **Training Platform**: Kaggle
* **Training Dataset**: [165 Different Snakes Species – Kaggle](https://www.kaggle.com/datasets/goelyash/165-different-snakes-species)
* **Training Epochs**: 100
* **Training Accuracy**: 66.1%
* **Export Formats**: `.pkl` (FastAI), `.onnx` (Opset 12)

---

## 🧠 Model Output

| Format          | Output           | Lookup Key in `species.csv` |
| --------------- | ---------------- | --------------------------- |
| `.pkl` (FastAI) | `class_id` (int) | `class_id`                  |
| `.onnx`         | `index` (int)    | `index`                     |

The `species.csv` file maps model output to rich metadata:

**`species.csv` Columns:**

* `index` – ONNX model output index (0–134)
* `class_id` – FastAI model output class ID
* `binomial_name` – Scientific name
* `common_name` – Common name (if available)
* `country`, `continent` – Geographic distribution
* `genus`, `family`, `snake_sub_family` – Taxonomic classification
* `poisonous` – Boolean indicating whether the species is venomous

---

## 📁 Files Included

| File Name     | Description                               |
| ------------- | ----------------------------------------- |
| `model.onnx`  | Trained model (ONNX format)               |
| `model.pkl`   | Trained model (FastAI format)             |
| `species.csv` | Species metadata and index/class mappings |

---

## 🛠️ Training Script

The model was trained using FastAI on Kaggle. Here's the full training script:

```python
from fastai.vision.all import *

dls = ImageDataLoaders.from_folder(
    '/kaggle/input/165-different-snakes-species',
    valid_pct=0.2,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(),
)

learn = vision_learner(dls, resnet101, metrics=accuracy)
learn.fine_tune(100)
learn.export('/kaggle/working/model.pkl')
```

---

## 🚀 Usage Examples

### 🔷 FastAI `.pkl` Model

```python
from fastai.vision.all import *
import pandas as pd

# Load the trained model
learn = load_learner('model.pkl')

# Load species metadata
species_df = pd.read_csv('species.csv')

# Load and predict on a single image
img_path = 'path/to/image'  # Replace with your image path
pred_class, pred_idx, probs = learn.predict(img_path)

# Match prediction using class_id (FastAI outputs class_id)
match = species_df[species_df['class_id'] == int(pred_class)].iloc()[0]

# Display full species info
print(f"🔍 Predicted class_id for image: {pred_class}")
print(f"Species: {match['common_name']} ({match['binomial_name']})")
print(f"Venomous: {match['poisonous'] == 1}")
print(f"Genus: {match['genus']}, Family: {match['family']}, Subfamily: {match['snake_sub_family']}")
print(f"Found in: {match['country']} ({match['continent']})")
```

---

### 🔶 ONNX Model

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd

# Load species metadata
species_df = pd.read_csv('species.csv')

# Define image preprocessing (similar to FastAI)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means
                         [0.229, 0.224, 0.225])  # ImageNet stds
])

# Load and preprocess the image
img_path = 'path/to/image'  # Replace with your image path
img = Image.open(img_path).convert("RGB")
img_tensor = preprocess(img).unsqueeze(0).numpy()  # Shape: [1, 3, 224, 224]

# Load ONNX model
ort_session = ort.InferenceSession('model.onnx')
input_name = ort_session.get_inputs()[0].name
outputs = ort_session.run(None, {input_name: img_tensor})

# Get predicted index
pred_idx = int(np.argmax(outputs[0]))  # ONNX model outputs class index (0–134)

# Match with species.csv using 'index' (not class_id)
match = species_df[species_df['index'] == pred_idx].iloc[0]

# Display full species info
print(f"\n🔍 Predicted index for image: {pred_idx}")
print(f"Species: {match['common_name']} ({match['binomial_name']})")
print(f"Venomous: {match['poisonous'] == 1}")
print(f"Genus: {match['genus']}, Family: {match['family']}, Subfamily: {match['snake_sub_family']}")
print(f"Found in: {match['country']} ({match['continent']})")
```

---

## 🛠️ Notes

* This is my **first machine learning project**, built as a foundation for a future real-time snake identification app.
* Accuracy is modest (66.1%), and the model is not yet production-grade.
* All feedback, improvements, and contributions are **welcome and appreciated**!

---

## 📬 Contact & Contributions

Feel free to fork the project, suggest improvements, or open issues. 

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

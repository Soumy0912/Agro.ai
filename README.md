<div align="center">
  <h1>🌿 Agro.ai: Plant Disease Detection System</h1>
  <p>An intelligent crop disease diagnostic tool leveraging deep learning to identify 38+ plant diseases and provide actionable treatments & supplement recommendations.</p>

  [![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 Overview

Agro.ai bridges the gap between agricultural practitioners and cutting-edge artificial intelligence. By simply uploading an image of a crop leaf, the system classifies the specific disease and immediately retrieves descriptive information, treatment steps, and relevant supplement purchasing links. 

The underlying AI pipeline utilizes **MobileNetV2** for robust feature extraction followed by a custom **Convolutional Neural Network (CNN)** for high-accuracy classification across 38 distinct plant disease categories.

---

## ✨ Key Features

- **Accurate Disease Detection:** Classifies 38 distinct crop conditions from normal health to severe diseases (e.g., Apple Scab, Tomato Leaf Mold).
- **Fast Feature Extraction:** Uses a pre-trained `MobileNetV2` model for highly efficient and accurate image feature vectorization.
- **Actionable Treatments:** Maps predicted diseases to dedicated CSV databases to retrieve immediate descriptions and mitigation steps.
- **Supplement Recommendations:** Directly suggests relevant chemical/organic supplements and provides direct buy links.
- **Cloud-Ready Interactive UI:** Provides a clean, responsive front-end built with `Streamlit` to easily upload and analyze images.

---

## 🏗️ Architecture & Models

### 1. Data Pipeline
- **Dataset:** ~87,000 RGB images sourced from the [PlantVillage / Kaggle Plant Disease Dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset).
- **Preprocessing:** Images are resized to `224x224`, converted to arrays, and preprocessed using MobileNetV2 standards.
- **Feature Extraction:** Extracts 1280-dimensional feature vectors per image using `MobileNetV2` (weights='imagenet', include_top=False).

### 2. Custom CNN Classifier
The classification layer processes the normalized 1280-D features through a custom Convolutional Neural Network architecture:
```python
Conv1D (64 filters, 3 kernel) + ReLU
MaxPooling1D (pool 2)
Conv1D (32 filters, 3 kernel) + ReLU
MaxPooling1D (pool 2)
Flatten -> Dense (128) + ReLU
Dropout (0.5)
Dense (38) + Softmax
```
*Note: We initially benchmarked against a Random Forest classifier, but the deep CNN approach yielded superior accuracy metrics.*

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Python 3.9+** and `pip` installed.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Yatin07/Agro_AI.git
   cd Agro_AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: The system requires `gdown` to automatically pull the pre-computed feature mapping CSV (`shuffled_file.csv`) from Google Drive on the first run.*

3. **Run the Streamlit Application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the App**
   Open your browser and navigate to `http://localhost:8501`.

---

## 🐳 Docker Support

To run the application inside a Docker container:

1. **Build the image**
   ```bash
   docker build -t agro-ai .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 agro-ai
   ```

---

## 📁 Repository Structure

```text
Agro_AI/
├── streamlit_app.py        # Main Streamlit application
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container setup
├── shuffuled_model.h5      # Custom trained CNN model
├── disease_info.csv        # Treatment procedures & descriptions
├── supplement_info.csv     # Supplement recommendations & links
└── packages.txt            # System-level dependencies
```

---

## 🤝 Contributing & Contact

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/Yatin07/Agro_AI/issues) if you want to contribute.

- **Author:** [Yatin Patil](https://github.com/Yatin07)

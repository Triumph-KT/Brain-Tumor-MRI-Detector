# Brain Tumor MRI Classification with Transfer Learning

Developed a deep learning model to classify MRI brain scans as either **Pituitary Tumor** or **No Tumor** using Convolutional Neural Networks (CNN) and **Transfer Learning** with the VGG16 architecture.

## Project Overview

This project applies advanced computer vision techniques to the detection of brain tumors from MRI images. By leveraging **Transfer Learning**, the model builds on pre-trained knowledge from large-scale datasets to improve accuracy and efficiency on a relatively small medical imaging dataset.

## Dataset

- **Source**: [Brain Tumor Classification MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- Total images: **1,000 MRI scans**
  - **395 No Tumor images**
  - **435 Pituitary Tumor images**
- Images preprocessed to size **224 x 224** pixels.
- Train-test split handled via directory structure with automatic labeling.

## Objectives

- Preprocess and augment MRI image data to enhance generalization.
- Build a baseline CNN model for binary classification.
- Improve performance with **Transfer Learning** using **VGG16**.
- Evaluate and compare model performance.
- Provide recommendations for further development and deployment in clinical settings.

## Methods

- **Data Preprocessing**:
  - Rescaling pixel values.
  - Automatic labeling with Keras `flow_from_directory`.
- **Data Augmentation**:
  - Horizontal flips.
  - Random rotations (up to 20°).
  - Height and width shifts.
  - Shear and zoom transformations.
- **Modeling**:
  - Baseline CNN with:
    - Multiple convolutional layers.
    - Batch Normalization and Dropout.
    - Sigmoid output for binary classification.
  - Transfer Learning with:
    - Pre-trained **VGG16** (frozen convolutional base).
    - Custom fully connected layers.
    - Binary output layer.
- **Training**:
  - CNN trained for **10 epochs**.
  - Transfer Learning model trained for **5 epochs**.
- **Evaluation**:
  - Accuracy on training and validation sets.
  - Visualization of accuracy over epochs.
  - Analysis of overfitting and convergence behavior.

## Results

- **Custom CNN**:
  - ~95% training accuracy.
  - Lower validation accuracy, with overfitting concerns.
- **VGG16 Transfer Learning**:
  - Achieved higher validation accuracy in **half the epochs**.
  - Demonstrated faster convergence and superior generalization.
- Transfer Learning significantly outperformed the baseline model despite a similar number of trainable parameters.

## Business/Scientific Impact

- Demonstrated the potential for rapid, automated brain tumor detection to support radiologists and medical professionals.
- Validated the use of **Transfer Learning** in medical imaging tasks with limited data.
- Provided a scalable, interpretable model architecture applicable to other healthcare image classification problems.

## Technologies Used

- Python
- TensorFlow (Keras)
- VGG16
- ImageDataGenerator
- Matplotlib
- NumPy
- Pandas

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/brain-tumor-classification.git
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open and run the notebook to:
   - Preprocess the dataset.
   - Train both the baseline CNN and Transfer Learning models.
   - Evaluate and compare model performance.

## Future Work

- Expand dataset with more diverse tumor types (e.g., Glioma, Meningioma).
- Implement advanced Transfer Learning techniques with fine-tuning.
- Integrate **data augmentation** with medical imaging domain knowledge.
- Explore additional architectures such as **ResNet** and **Inception**.
- Prepare the model for deployment in clinical decision support systems.

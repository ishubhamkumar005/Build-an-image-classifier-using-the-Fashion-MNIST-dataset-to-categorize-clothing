# Build-an-image-classifier-using-the-Fashion-MNIST-dataset-to-categorize-clothing
# Image-Classifier-for-Fashion-MNIST-Dataset

# 🧠 Image Classifier for Fashion MNIST Dataset

This project is an **image classification solution** using the **Fashion MNIST** dataset. We leverage **Convolutional Neural Networks (CNNs)** built with **TensorFlow** and **Keras** to accurately categorize grayscale images of clothing items into **10 distinct classes**. Additionally, we analyze the classifier’s performance using a **confusion matrix** for a comprehensive evaluation.

---

## 📌 Problem Statement

The **Fashion MNIST dataset** provides a more challenging alternative to the classic MNIST dataset by offering grayscale images of fashion articles instead of handwritten digits.  
Our **goal** is to:

✅ **Build an image classifier** capable of recognizing different categories of clothing.  
✅ **Achieve high accuracy** on the unseen test dataset.  
✅ **Analyze the model’s performance** by visualizing the confusion matrix to understand misclassifications and identify areas for further improvement.

---

## 📂 Dataset Overview

- **Dataset:** [Fashion MNIST on Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)  
- **Description:** Contains 70,000 grayscale images (28x28 pixels) of 10 fashion categories:
  - `0`: T-shirt/top
  - `1`: Trouser
  - `2`: Pullover
  - `3`: Dress
  - `4`: Coat
  - `5`: Sandal
  - `6`: Shirt
  - `7`: Sneaker
  - `8`: Bag
  - `9`: Ankle boot

- **Split:**
  - **Training set:** 60,000 images  
  - **Test set:** 10,000 images  

---

## 🧪 Solution Approach

Our approach involves several well-defined steps:

### 1️⃣ Data Loading
We use TensorFlow’s built-in `datasets.fashion_mnist.load_data()` method to load the dataset into:
- **Training images & labels**: `(train_images, train_labels)`
- **Test images & labels**: `(test_images, test_labels)`

### 2️⃣ Data Preprocessing
- **Normalization:**  
  Pixel values originally range from 0 to 255. We **scale them to [0, 1]** for faster convergence during training:

  ```python
  train_images = train_images / 255.0
  test_images = test_images / 255.0
Reshaping:
The CNN expects a channel dimension (channels=1 for grayscale).
Reshape images to (28, 28, 1)

3️⃣ Model Architecture
We design a Convolutional Neural Network (CNN) as follows:

Layer 1: Conv2D with 32 filters, kernel size 3x3, ReLU activation, input shape (28, 28, 1)

Layer 2: MaxPooling2D with pool size 2x2

Layer 3: Conv2D with 64 filters, kernel size 3x3, ReLU activation

Layer 4: MaxPooling2D with pool size 2x2

Layer 5: Flatten to convert 2D feature maps to 1D

Layer 6: Dense layer with 64 units, ReLU activation

Output Layer: Dense layer with 10 units (for 10 classes, no activation since we’ll apply from_logits=True)

This design captures local spatial features in the images and progressively learns hierarchical patterns.

4️⃣ Model Compilation
We compile the model with:

Optimizer: Adam for adaptive learning.

Loss: SparseCategoricalCrossentropy with from_logits=True since we do not apply softmax in the last layer.

Metric: Accuracy to monitor classification performance.

5️⃣ Model Training
We train the model for 10 epochs with 10% of training data as validation to monitor overfitting and fine-tune hyperparameters.

python
Copy
Edit
model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

6️⃣ Model Evaluation
After training, we evaluate the classifier on the test dataset
This provides an unbiased measure of the model’s ability to generalize to unseen data.

7️⃣ Predictions & Probability
Since the last Dense layer does not include softmax, we append it during prediction
We then compute the predicted labels:
predicted_labels = np.argmax(predictions, axis=1)

8️⃣ Confusion Matrix Visualization
To better understand misclassifications, we compute and plot a confusion matrix.
The confusion matrix highlights where the classifier struggles, e.g., confusing similar-looking items (like shirts and T-shirts).


📈 Key Takeaways:
✅ CNNs are highly effective for image classification tasks.

✅ Normalization and reshaping are critical to ensure proper data format for CNN input.

✅ Confusion matrices offer valuable insights into model errors and areas for improvement.

✅ This project forms a foundation for future enhancements such as:

Hyperparameter tuning

Data augmentation

Transfer learning using pre-trained models

📎 References:
Dataset: Fashion MNIST on Kaggle

TensorFlow Documentation: https://www.tensorflow.org/api_docs

Keras API: https://keras.io/api/

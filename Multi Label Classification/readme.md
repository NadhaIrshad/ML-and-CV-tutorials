# Dog Breed Classification Using CNN and Transfer Learning

This project aims to classify dog breeds from images using convolutional neural networks (CNN) and transfer learning with pre-trained models like Xception. The solution includes data preprocessing, model training, hyperparameter tuning, fine-tuning, and a prediction interface.

---

## Project Overview

The goal is to classify dog breeds from images using deep learning. The dataset consists of images of 15 popular dog breeds in France (2020). Data augmentation techniques were used to increase dataset size and diversity.

---

## Data Preparation

- Images were preprocessed by resizing, denoising, and histogram equalization.  
- Data augmentation included random zoom, rotation, and horizontal flipping.  
- Labels were encoded numerically using `LabelEncoder`.  
- Training and test splits were created, with validation handled by Keras generators.

---

## Models

- **Baseline CNN from Scratch:** A simple CNN with 3 convolutional layers was trained as a baseline but achieved limited accuracy (~13%).  
- **Transfer Learning with Xception:** Leveraged the pre-trained Xception model without the fully connected layers. Added a custom classifier, froze base layers, and trained only the classifier.  
- **Comparison with ResNet50:** Tested but performed worse than Xception in this task.

---

## Hyperparameter Tuning

Used [Keras Tuner](https://keras.io/keras_tuner/) to optimize parameters of the classifier added on top of Xception, improving model performance before fine-tuning.

---

## Fine-Tuning

Unfroze the last block of the Xception model and retrained it along with the classifier to improve the model’s accuracy and robustness.

---

## Evaluation

The fine-tuned Xception model was evaluated on the test dataset using metrics like accuracy and F1-score, showing strong performance and good generalization.

---

## Prediction Interface

Implemented a user-friendly interface using [Gradio](https://gradio.app/) to allow users to upload an image and receive a breed prediction in real-time.

---

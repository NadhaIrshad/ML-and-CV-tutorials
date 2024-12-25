## Object Detection using Python and TensorFlow

This project implements an object detection pipeline using Python and TensorFlow, from image acquisition to model training and final inference on both images and videos. The implementation leverages several libraries and tools for data preprocessing, augmentation, model design, and results visualization.

---

## **Features**

- **Image Acquisition**: Captures images using OpenCV for dataset preparation.  
- **Data Annotation**: Annotates images using [Labelme](https://github.com/wkentaro/labelme) to create labeled datasets.  
- **Data Augmentation**: Applies transformations using the powerful [Albumentations](https://albumentations.ai/) library to enhance dataset diversity.  
- **Model Training**: Implements a custom object detection model using TensorFlow's Functional API and a pre-trained VGG16 backbone.  
- **Inference and Visualization**: Processes a test image and video using OpenCV to showcase the detection results.  

---

## **Setup and Installation**

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- OpenCV
- Labelme
- Albumentations
- Matplotlib
- NumPy



## **Steps Involved**

1. **Image Acquisition**  
   Images were collected using OpenCV to create a dataset for training and testing.

2. **Annotation**  
   The images were annotated using Labelme to generate the corresponding JSON files for bounding boxes and class labels.

3. **Data Augmentation**  
   Augmented the annotated dataset using Albumentations to increase its size and improve model robustness. Augmentations included transformations like flipping, rotation, and brightness adjustment.

4. **Model Training**  
   - Used TensorFlow's Functional API to build a deep neural network.  
   - Integrated VGG16 as a feature extractor to leverage pre-trained weights for better performance.  
   - Trained the network on the annotated and augmented dataset.

5. **Inference**  
   - Tested the trained model on new data.  
   - Visualized the results on both images and video using OpenCV.

---


## **Results**

- The model effectively detects objects in test images and videos.  
- Detection results are displayed using OpenCV, with bounding boxes and class labels.  


Sample Results:
- **Imag![Object Detection 12_25_2024 10_11_40 PM](https://github.com/user-attachments/assets/a9be377f-6449-4077-99b9-9a216796ad89)
e Detection**:
 
- **Video Detection**:
  
https://github.com/user-attachments/assets/26878efc-9aac-40c9-8708-d5494d1be9b8


---


## **Acknowledgments**
- Nicholas Renotte YT channel https://youtu.be/N_W4EYtsa10?si=cNU2TJLwiRy0R9e9


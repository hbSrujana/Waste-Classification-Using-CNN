# Week1
Plastic Waste Classification using CNN: Week 1 work for the Edunet-Shell Skills4Future AICTE Internship, including dataset download, preprocessing, and label visualization to classify plastic waste into organic and recyclable categories.

### Dataset
The `dataset` folder contains two subfolders:  
- **TRAIN**: For training the model.  
- **TEST**: For testing the model.

### Tools and Libraries
- Python  
- TensorFlow and Keras  
- OpenCV for image processing  
- Matplotlib for visualization  
- Pandas for data handling  

## Week 1 Work
- Downloaded the dataset and organized it into `TRAIN` and `TEST` folders.  
- Created the file `waste_classification.ipynb` for model development.  
- Preprocessed the dataset:
  - Loaded images using OpenCV and converted them to RGB format.  
  - Created a DataFrame with images and their respective labels.  
  - Visualized label distribution using pie charts.
    
---

## Week 2 - Convolutional Neural Network (CNN) for Image Classification
In Week 2, we focused on building a CNN model using TensorFlow and Keras to classify images. The dataset consists of labeled images, and we performed data preprocessing, model training, and evaluation.

## Implemented Features
- **Data Preprocessing**: Used ImageDataGenerator to rescale images and load them from directories.

- **Model Architecture**:
  - Three convolutional layers with ReLU activation and max-pooling.
  - Fully connected layers with dropout for regularization.
  - Final output layer using softmax activation for binary classification.

- **Training Process**:
  - Compiled the model using Adam optimizer and binary cross-entropy loss.
  - Trained for 15 epochs with a batch size of 64.
  - Validated performance using a test dataset.

- **Visualization**:
  Randomly displayed sample images with their corresponding labels.
  
## Model Architectre
  The model consists of the following layers:
  - Conv2D (32 filters, 3x3 kernel, ReLU activation, MaxPooling)
  - Conv2D (64 filters, 3x3 kernel, ReLU activation, MaxPooling)
  - Conv2D (128 filters, 3x3 kernel, ReLU activation, MaxPooling)
  - Flatten layer
  - Fully Connected Layer (256 neurons, ReLU, Dropout 0.5)
  - Fully Connected Layer (64 neurons, ReLU, Dropout 0.5)
  - Output Layer (2 neurons, softmax activation for binary classification)

## Training Details
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Batch Size**: 64
- **Epochs**: 15
- **Data Augmentation**: Rescaling of pixel values

## Results
Model training was successfully completed with validation on the test dataset.
The trained model can classify images into two categories.

---

# Week3
  In Week 3 of my internship, I worked on evaluating the performance of our CNN model, implementing a prediction function for waste classification, and integrating a Streamlit-based web application for real-time classification. The model is trained to classify images into Recyclable or Organic Waste.
  
## Implemented Features
- **Performance Visualization**:
  - Plotted training vs validation accuracy.
  - Plotted training vs validation loss.

- **Prediction Function**:
  - Implemented predict_fun(img) to classify images as either Recyclable or Organic Waste.
  - Uses OpenCV for image loading and preprocessing.
  - Utilizes the trained CNN model for classification.

- **Testing the Model**:
  - Loaded test images from the dataset.
  - Used predict_fun() to classify test images.

- **Streamlit Application**:
  - Developed Waste_classification.py using Streamlit for an interactive web-based waste classification system.
 - Users can upload images, and the model will predict whether the waste is Recyclable or Organic.

## Results & Observations
- he model performs well in classifying waste categories.
- The loss and accuracy plots indicate training stability.
- Some misclassifications suggest further tuning is required.
- The Streamlit application provides an easy-to-use interface for real-time classification.

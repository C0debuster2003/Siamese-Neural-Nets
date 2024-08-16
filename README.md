# Siamese Neural Networks for Facial Recognition
# Overview
This project implements a facial recognition system using Siamese Neural Networks (SNNs). Siamese networks are particularly well-suited for one-shot learning tasks, such as facial recognition, where the goal is to distinguish between two inputs based on their similarity. In this case, the network learns to differentiate between facial images by comparing them and outputting a similarity score.

# Features
Siamese Neural Network Architecture: Utilizes twin networks with shared weights to process input pairs.
Face Verification: Given two images, the model predicts whether they belong to the same person.
One-shot Learning: Capable of learning new faces with very few examples.
Customizable: Easily adaptable for different image sizes and architectures.
Preprocessing: Includes face detection and alignment as preprocessing steps.
# Installation Prerequisites
Python 3.7+
TensorFlow
OpenCV
NumPy
Matplotlib
Scikit-learn
(Optional) CUDA for GPU support

# For model training, I have outlined the necessary instructions in the Jupyter notebook itself. 
If the performance is too slow, then try switching of the Data Augmentation layers as they themselves are quite compute heavy to run

# Inference
You can also use the trained model for inference on new image pairs. Just pass in an Anchor image and click an image through the Kivy application for the other image. Depending on it,
it will be categorized as a positive or a negative image, positive if the image matches the anchor image over a certain threshold and negative if it is below the threshold.
The threshold is also user-specified. 
Also for verification, the captured image is not matched with a sinle anchor image. Rather, it is matched will all the anchor images in this implementation. The reason is to improve
the model robustness by reducing the number of false positives.

# Results
The model achieves an accuracy of approximately 97%. The accuracy may vary depending on the architecture and dataset used.

# Customization
Network Architecture: The current implementation uses a convolutional neural network (CNN) as the base model. You can easily replace it with more complex architectures like ResNet or MobileNet.
Loss Function: By default, contrastive loss is used to train the Siamese network. You can experiment with other loss functions like triplet loss.
Image Preprocessing: The current pipeline includes basic preprocessing steps. You can add more advanced preprocessing like histogram equalization, data augmentation, etc.
Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes or improvements.

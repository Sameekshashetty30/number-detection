# number-detection

📌 Project 1: Handwritten Digit Recognition (MNIST)
Overview
This project uses a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. It also includes preprocessing of a custom image (6.jpeg) to test the model's prediction capabilities. 
Highlights
- Preprocessing pipeline for grayscale digit images
- CNN architecture with dropout and max pooling
- Training on MNIST dataset with validation
- Prediction on custom input image
- Visualization of original and preprocessed image 
Model Architecture
- Conv2D → ReLU
- Conv2D → ReLU
- MaxPooling2D
- Dropout
- Flatten
- Dense → ReLU
- Dropout
- Dense → Softmax
How to Run
python mnist_digit_classifier.py



🌸 Project 2: Iris Flower Classification
Overview
This project uses a simple feedforward neural network to classify Iris flowers into three species based on four features: sepal length, sepal width, petal length, and petal width.
Highlights
- One-hot encoding of target labels
- Dense neural network with softmax output
- Training and validation accuracy tracking
- Confusion matrix visualization
- Training history plots for accuracy and loss
Model Architecture
- Dense → ReLU
- Dense → Softmax
How to Run
python iris_flower_classifier.py



📊 Visualizations
- Confusion matrix for Iris predictions
- Accuracy and loss curves for training and validation
- Input image visualization for MNIST digit prediction

🛠 Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- OpenCV
- scikit-learn
Install dependencies using:
pip install -r requirements.txt



🚀 Future Improvements
- Add model saving/loading functionality
- Extend MNIST model to support more custom inputs
- Experiment with deeper architectures and hyperparameter tuning
- Integrate Grad-CAM for interpretability (especially for MNIST)


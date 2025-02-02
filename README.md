MNIST Neural Network

This project is a simple neural network built from scratch to classify handwritten digits from the MNIST dataset. The neural network is implemented using basic Python libraries and doesn't rely on pre-built deep learning frameworks such as TensorFlow or PyTorch.

Project Overview

The goal of this project is to build and train a neural network from scratch to recognize digits in the MNIST dataset. This dataset contains 60,000 training images and 10,000 testing images of handwritten digits (0-9), each 28x28 pixels in size.

  Key Features:
- Neural Network Built From Scratch: The neural network is implemented using only Python libraries, without relying on deep learning frameworks.
- Training with Backpropagation: The network is trained using backpropagation and stochastic gradient descent (SGD) for weight updates.
- MNIST Dataset: The network is trained and tested on the MNIST dataset, which is widely used for benchmarking machine learning algorithms.
  
Installation

1. Clone this repository:
   git clone https://github.com/hrishkul/MNIST-neural-network.git
   
2. Navigate to the project directory:
   cd MNIST-neural-network
   
3. Install the required dependencies:
   pip install -r requirements.txt

4. Usage
   To train and test the neural network, run the mnist-neural-network.py script:
   python mnist-neural-network.py
   This will load the MNIST dataset, train the neural network, output the accuracy on the test set after training is complete and predict some examples from the test set.

Architecture
The neural network consists of:

Input Layer: 784 input neurons (one for each pixel of the 28x28 images).
Hidden Layer(s): Customizable number of hidden layers with ReLU activation functions.
Output Layer: 10 output neurons, one for each possible digit (0-9), with a softmax activation function.
The network is trained using stochastic gradient descent (SGD) with backpropagation to minimize the cross-entropy loss.

Results
After training, the neural network achieves a high classification accuracy of upto 84.5 % on the MNIST training set, demonstrating the power of a simple neural network built from scratch.
Made 10 predictions on the test data and achieved a high accuracy.
Final test result screenshots are attached in the images folder.


Contributing
Feel free to fork this repository and submit pull requests for improvements or suggestions!

License
This project is licensed under the MIT License - see the LICENSE file for details.

   
   

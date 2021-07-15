# AdversarialDefense

This repo contains the code produced by Shrey Poshiya during the summer 2021 internship at the [Institute for Computing in Research](https://computinginresearch.org/). 

License: [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

### What are Adversarial Examples

Recent developments in Deep Learning ([DL](https://en.wikipedia.org/wiki/Deep_learning)) has allowed for its implementation into a wide array of applications. With deep learning being used in many saftey critical environments (ex. [healthcare](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6945006/) and [transporation](https://mobility.mit.edu/machine-learning)), it is becoming increasingly important that these aritifical neural networks can succesfully identify the given inputs.

It is been found that carefully altered inputs, called [adversarial examples](https://arxiv.org/abs/1412.6572) can trick neural networks. These adversarial examples are usually produced by intentionally adding noise to the input image. The most dangerous aspect of these adversarial examples is that the difference between the original input image and the altered one is virtually imperceptable to the human eye. The combination of being able to fool the network while seeming harmless to the human poses a great danger to the validity of the neural nets. 

Here is an example of an adversraial example:
![image](https://user-images.githubusercontent.com/86625362/125666483-09db1541-f6f7-4597-b9f0-56f779ba0a40.png)

### How are Adversarial Examples Generated

One popular strategy to produce adversarial examples is uing the Fast Sign Gradient Method (FSGM). This strategy exploits the gradient, a numeric calculation that gives us information on how to adjust the parameters of the model to minmize the deviation between the actual output and the output estimated by the network. The gradient is a vector that signals in which direction the loss in the [loss function](https://en.wikipedia.org/wiki/Loss_function) increases. In order to create a good neural network (in which the loss is minimized), we sould move in the opposite direction of the gradient and change the parameters of the model in accordance (esentailly we are finding the local minimum of the loss function). The Fast Sign Gradient Method uses the generated gradients to create an image that maximizes the loss/cost.


![image](https://user-images.githubusercontent.com/86625362/125686603-1c5dfa98-3185-4515-84d4-e5be6c0e14e6.png)

![image](https://user-images.githubusercontent.com/86625362/125686712-40e1a999-8f47-47bb-b6ca-aad42458e2eb.png): Our output adversarial image 

![image](https://user-images.githubusercontent.com/86625362/125686803-f3348a3a-50df-4119-bdc0-f9ce7b07d34d.png): Original Input Image

![image](https://user-images.githubusercontent.com/86625362/125686859-85448ff8-4f98-43fa-86de-ab864ab41f72.png): The label of the Input Image

![image](https://user-images.githubusercontent.com/86625362/125686911-7bdd529f-f2f9-4df6-b8ae-2408e17a087b.png): A constant that dictates the intensity of the perturbations (the noise) applied to the input image

![image](https://user-images.githubusercontent.com/86625362/125687293-b09136fb-42bb-4f04-a8cd-ed0cad3c3f16.png): The Neural Network model

![image](https://user-images.githubusercontent.com/86625362/125687333-06adcf8e-d33d-4ac3-8d39-ca6821062fb2.png): The Loss

### Contents of this Repo

This repo contains three different folders. Each folder contains scripts for three different Neural Networks (the folder name specifies the name of the dataset that specfic network is using: [MNIST](http://yann.lecun.com/exdb/mnist/), [fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)). 

The folder has the following contents:

**{NameOfDataset}.py** = this script creates a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) and train the network using [tensorflow.keras](https://keras.io/about/)

**Adv_Gen_{NameOfDataset}.py** = this scripts generates adversarial images at a given epsilon value. Uses the generated adversarial images to test accuracy of the model when the adverarial images are fed in.

**blur_{NameOfDataset}.py** = This script blurs the generated adverarial images. This should esentially "blur" out the perturbations in the image. The accuracy when the blurred adverarial images are used to test the accuracy of the model should be higher than when the adversarial images are used to test the accuracy (In fact the accuracy of the original dataset versus the blurred adversarial iamges was very close when a certain amount of "blurring" was applied. In my tests the discrpancy between the two accuracies was only -0.13% to -0.25%). 


![image](https://user-images.githubusercontent.com/86625362/125684261-121ec6a9-114b-4149-8e0e-fcfa4b622c9d.png)

Still yet to create section about strategies to combat adverarial attacks














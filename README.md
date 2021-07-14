# AdversarialDefense

This repo contains the code produced by Shrey Poshiya during the summer 2021 internship at the [Institute for Computing in Research](https://computinginresearch.org/). 

### What are Adversrial Examples

Recent developments in Deep Learning (DL) has allowed for its implementation into a wide array of applications. With deep learning being used in many saftey critical environments (healthcare, transortation, and the military sector), it is becoming increasingly important that these aritifical neural networks can succesfully identify their given inputs. 

It is been found that carefully altered inputs, called *adversarial examples* can trick neural networks. These adversarial examples are usually produced by adding a minsucle amount of noise to the input image. But the most dangerous aspect of these adversarial examples is that the difference between the original input image and the altered one is virtually imperceptable to the human eye. This combination of being able to fool the neural net while eeming harmless to the human poses a great danger to the validity of the neural nets. 

Here is an example of an adversraial example:
![image](https://user-images.githubusercontent.com/86625362/125666483-09db1541-f6f7-4597-b9f0-56f779ba0a40.png)
![image](https://user-images.githubusercontent.com/86625362/125684261-121ec6a9-114b-4149-8e0e-fcfa4b622c9d.png)

### How are Adversarial Examples Generated

One of the most popular strategies to produce adversarial examples is uing the Fast Sign Gradient Method (FSGM). This strategy exploits the gradient, a numeric calculation that gives us information on how to adjust the parameters of the model to minmize the deviation between the actual output and the one estimated by the network. If you take the loss funciton of the model. So by moving in the opposite direction of the gradient we can generate an image that minimize the deviation between the two outputs (the actual output and the output estimated by the network).

![image](https://user-images.githubusercontent.com/86625362/125686603-1c5dfa98-3185-4515-84d4-e5be6c0e14e6.png)

![image](https://user-images.githubusercontent.com/86625362/125686712-40e1a999-8f47-47bb-b6ca-aad42458e2eb.png): Our output adversarial image 

![image](https://user-images.githubusercontent.com/86625362/125686803-f3348a3a-50df-4119-bdc0-f9ce7b07d34d.png): Original Input Image

![image](https://user-images.githubusercontent.com/86625362/125686859-85448ff8-4f98-43fa-86de-ab864ab41f72.png): The label of the Input Image

![image](https://user-images.githubusercontent.com/86625362/125686911-7bdd529f-f2f9-4df6-b8ae-2408e17a087b.png): A number that dictates the intensity of the perturbations (the noise) applied to the input image

![image](https://user-images.githubusercontent.com/86625362/125687293-b09136fb-42bb-4f04-a8cd-ed0cad3c3f16.png): The Neural Network model

![image](https://user-images.githubusercontent.com/86625362/125687333-06adcf8e-d33d-4ac3-8d39-ca6821062fb2.png): The Loss















import Cifar10
import numpy
import tensorflow as tf
from Cifar10 import x_test, y_test, model
import matplotlib.pyplot as plt


######################
#ADVESARAIAL EXAMPLES#
######################


def advesarial(image, label):

  image = tf.cast(image, tf.float32)

  with tf.GradientTape() as tape:
      tape.watch(image)
      prediction = model(image)
      loss = tf.keras.losses.MSE(label, prediction)
  
  gradient = tape.gradient(loss, image)

  signed_grad = tf.sign(gradient)

  return signed_grad



rg = [*range(0, x_test.shape[0],1)]
epsilons = [0, 0.05 , 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
acc = []

for numbers in epsilons: # reset value before each iteration 
  z = 0
  for i in range(len(rg)):
    image = x_test[i]
    image_label = y_test[i]
    #Create perturbations
    
    perturbations = advesarial(image.reshape(1, 32, 32, 3), image_label).numpy()

    # apply perturbations with epsilon(numbers)
    
    adv_img = image + perturbations * numbers
       

    # go through each produced image and put it through the model to
    # get prediction

    adv_pred = model.predict(numpy.array([adv_img.reshape(32,32,3)]))

    list_index = [0,1,2,3,4,5,6,7,8,9]
    x = adv_pred

    for i in range(10):
      for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
          temp = list_index[i]
          list_index[i] = list_index[j]
          list_index[j] = temp


    #Accuracy data 
    if list_index[0] == image_label: 
      z = z + 1
 
  accuracy = z / 10000
  print("epsilon value:", numbers , 'acc:', accuracy)
  acc.append(accuracy)

print(acc)

# Generate plot for advesarial data
plt.plot(epsilons, acc)
plt.title("accuracy vs epsilon")
plt.xlabel('epsilon value')
plt.ylabel('accuracy')
plt.show()
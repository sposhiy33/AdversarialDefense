import fashion_MNIST
import numpy
import tensorflow as tf
from fashion_MNIST import x_test, y_test, model
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
    #Create adversarial examples   
    perturbations = advesarial(image.reshape(1, 28, 28, 1), image_label).numpy()
    adv_img = image + perturbations * numbers
    # Get prediction of adversarial image
    adv_pred = model.predict(numpy.array([adv_img.reshape(28,28,1)]))
    # compile prediction
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

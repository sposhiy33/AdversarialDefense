from Cifar10 import x_test, y_test, y_test_one_hot, model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage


# function to generate adverarial examples
def advesarial(image, label):
  image = tf.cast(image, tf.float32)
  with tf.GradientTape() as tape:
      tape.watch(image)
      prediction = model(image)
      loss = tf.keras.losses.MSE(label, prediction)
  gradient = tape.gradient(loss, image)
  signed_grad = tf.sign(gradient)
  return signed_grad

#function that predicts the inputed image
def predict(image):
  image = image.reshape(1, 32, 32, 3)
  predictions = model.predict(np.array(image))
  list_index = [0,1,2,3,4,5,6,7,8,9]
  x = predictions
  for i in range(10):
    for j in range(10):
      if x[0][list_index[i]] > x[0][list_index[j]]:
        temp = list_index[i]
        list_index[i] = list_index[j]
        list_index[j] = temp
  return list_index

print("baseline accuracy:", model.evaluate(x_test, y_test_one_hot)[1])

ep = 0.2
size = [1,2,3,4,5]
# number of images
rg = [*range(0, x_test.shape[0],1)]



## MAXIMUM FILTER
accmax = []
for elements in size:
    z = 0
    for i in range(len(rg)):
        img = x_test[i]
        img_label = y_test[i]
        # generate adverarial image
        perturbations = advesarial(img.reshape(1, 32, 32, 3), img_label).numpy()
        adv_img = img + perturbations * ep
        # Apply maximum filter
        max = ndimage.maximum_filter(adv_img, size = elements)
            
        ### MAX PRED
        max_pred = model.predict(np.array(max))
        list_index = [0,1,2,3,4,5,6,7,8,9]
        x = max_pred
        for i in range(10):
            for j in range(10):
                if x[0][list_index[i]] > x[0][list_index[j]]:
                    temp = list_index[i]
                    list_index[i] = list_index[j]
                    list_index[j] = temp
  
        # update accuracy countre if correct
        if list_index[0] == img_label:
            z = z + 1

    # calculate final accuracy by dividing the accruacy counter by the total number of images in the test dataset
    accuracy = z / 10000
    print("ep value:", ep, "size:", elements, 'acc:', accuracy)
    # print the list with all the produced accuracy values at different sigma values
    accmax.append(accuracy)
print("MAXPRED - epsilon value:", ep, "acc:", accmax)


## MINIMUM FILTER
accmin = []
for elements in size:
    z = 0
    for i in range(len(rg)):
        img = x_test[i]
        img_label = y_test[i]
        # generate adverarial image
        perturbations = advesarial(img.reshape(1, 32, 32, 3), img_label).numpy()
        adv_img = img + perturbations * ep
        # Apply maximum filter
        min = ndimage.minimum_filter(adv_img, size = elements)
            

        ### MIN PRED
        min_pred = model.predict(np.array(min))
        list_index = [0,1,2,3,4,5,6,7,8,9]
        x = min_pred
        for i in range(10):
            for j in range(10):
                if x[0][list_index[i]] > x[0][list_index[j]]:
                    temp = list_index[i]
                    list_index[i] = list_index[j]
                    list_index[j] = temp

        if list_index[0] == img_label:
            z = z + 1

    # calculate final accuracy by dividing the accruacy counter by the total number of images in the test dataset
    accuracy = z / 10000
    print("ep value:", ep, "size:", elements, 'acc:', accuracy)
    # print the list with all the produced accuracy values at different sigma values
    accmin.append(accuracy)
print("MINPRED - epsilon value:", ep, "acc:", accmin)


###MEDIAN FILTER
accmed = []
for elements in size:
    z = 0
    for i in range(len(rg)):
        img = x_test[i]
        img_label = y_test[i]
        # generate adverarial image
        perturbations = advesarial(img.reshape(1, 32, 32, 3), img_label).numpy()
        adv_img = img + perturbations * ep
        # Apply maximum filter
        med = ndimage.median_filter(adv_img, size = elements)

        ### MED PRED   
        med_pred = model.predict(np.array(med))
        list_index = [0,1,2,3,4,5,6,7,8,9]
        x = med_pred
        for i in range(10):
            for j in range(10):
                if x[0][list_index[i]] > x[0][list_index[j]]:
                    temp = list_index[i]
                    list_index[i] = list_index[j]
                    list_index[j] = temp

        if list_index[0] == img_label:
            z = z + 1

    # calculate final accuracy by dividing the accruacy counter by the total number of images in the test dataset
    accuracy = z / 10000
    print("ep value:", ep, "size:", elements, 'acc:', accuracy)
        # print the list with all the produced accuracy values at different sigma values
    accmed.append(accuracy)
print("MEDPRED - epsilon value:", ep, "acc:", accmed)

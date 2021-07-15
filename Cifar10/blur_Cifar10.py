# Libraries
import Cifar10
from Cifar10 import x_test, y_test, y_test_one_hot, model
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight") 


def advesarial(image, label):

  image = tf.cast(image, tf.float32)

  with tf.GradientTape() as tape:
      tape.watch(image)
      prediction = model(image)
      loss = tf.keras.losses.MSE(label, prediction)
  
  gradient = tape.gradient(loss, image)

  signed_grad = tf.sign(gradient)

  return signed_grad



############
#apply blur#
############


from scipy import ndimage

rg = [*range(0, x_test.shape[0],1)]

#blur coefficent (sigma)
sig = [0, 0.2, 0.4, 0.6, 0.8 , 1, 1.2, 1.4,]
sig_zoom = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
acc = []

print("baseline accuracy:", model.evaluate(x_test, y_test_one_hot)[1])

#0.15 ep value

for elements in sig:
        # reset accuracy counter every interation 
    z = 0 
    
    for i in range(len(rg)):
                
        image = x_test[i]
        image_label = y_test[i]
                
                #Create advesarial images
        perturbations = advesarial(image.reshape(1, 32, 32, 3), image_label).numpy()
        adv_img = image + perturbations * 0.2

        blur = ndimage.gaussian_filter(adv_img.reshape(32,32,3), sigma = elements)
                
        blur_pred = model.predict(np.array([blur.reshape(32,32,3)]))
                
        list_index = [0,1,2,3,4,5,6,7,8,9]
        x = blur_pred

        for i in range(10):
            for j in range(10):
                if x[0][list_index[i]] > x[0][list_index[j]]:
                    temp = list_index[i]
                    list_index[i] = list_index[j]
                    list_index[j] = temp  
            
        if list_index[0] == image_label:
            z = z + 1
                    
    accuracy = z / 10000
    print("blur coefficent:", elements, 'acc:', accuracy)
    acc.append(accuracy)
print("epsilon value: 0.2 acc:", acc)
        # Generate plot for advesarial data
plt.plot(sig, acc)
plt.title("accuracy vs sigma")
plt.xlabel('sigma value')
plt.ylabel('accuracy')
plt.show()

# Libraries
import Cifar10
from Cifar10 import x_test, y_test, y_test_one_hot, model
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight") 
from scipy import ndimage


def advesarial(image, label):

  image = tf.cast(image, tf.float32)

  with tf.GradientTape() as tape:
      tape.watch(image)
      prediction = model(image)
      loss = tf.keras.losses.MSE(label, prediction)
  
  gradient = tape.gradient(loss, image)

  signed_grad = tf.sign(gradient)

  return signed_grad


print("baseline accuracy:", model.evaluate(x_test, y_test_one_hot)[1])

rg = [*range(0, x_test.shape[0],1)]
epsilons = [0.0, 0.15, 0.20, 0.25, 0.30 ]
percent = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
percent_zoom = []
ep = 0.2


    
acc = []
    
for elements in percent:
            # reset accuracy counter every interation 
    z = 0 
        
    for i in range(len(rg)):
                    
        image = x_test[i]
        image_label = y_test[i]
                    
        #Create advesarial images
        perturbations = advesarial(image.reshape(1, 32, 32, 3), image_label).numpy()
        adv_img = image + perturbations * ep
            
        #apply percentile filter to the adversial images
        perc = ndimage.percentile_filter(adv_img, percentile = elements, size = 2)

        ### BLUR PRED       
        perc_pred = model.predict(np.array([perc.reshape(32,32,3)]))      
        list_index = [0,1,2,3,4,5,6,7,8,9]
        x = perc_pred
        for i in range(10):
            for j in range(10):
                if x[0][list_index[i]] > x[0][list_index[j]]:
                    temp = list_index[i]
                    list_index[i] = list_index[j]
                    list_index[j] = temp  
                
        if list_index[0] == image_label:
            z = z + 1   
                        
    accuracy = z / 10000
    print("ep value:", ep, "percent value:", elements, 'acc:', accuracy)
    acc.append(accuracy)
print("epsilon value:", ep, "acc:", acc)
# Generate plot for advesarial data
plt.plot(percent, acc)
plt.title("accuracy vs percent")
    # plt.legend(["epsilon value: 0.0", "epsilon value: 0.15", "epsilon value: 0.20", "epsilon value: 0.25", "epsilon value: 0.30"])
plt.xlabel('percent')
plt.ylabel('accuracy')
plt.show()

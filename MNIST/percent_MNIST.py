# Libraries
import MNIST
from MNIST import x_test, y_test, y_test_one_hot, model
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight") 
from scipy import ndimage
from scipy.interpolate import InterpolatedUnivariateSpline


def advesarial(image, label):
  image = tf.cast(image, tf.float32)
  with tf.GradientTape() as tape:
      tape.watch(image)
      prediction = model(image)
      loss = tf.keras.losses.MSE(label, prediction)
  gradient = tape.gradient(loss, image)
  signed_grad = tf.sign(gradient)
  return signed_grad

def generate_adverarial(img , img_label, epsilon):
    perturbations = advesarial(img.reshape(1, 28, 28, 1), img_label).numpy()
    global adv_img
    adv_img = img + perturbations * epsilon
    return adv_img 

def predict(im):
    blur_pred = model.predict(np.array([im])) 
    global list_index     
    list_index = [0,1,2,3,4,5,6,7,8,9]
    x = blur_pred
    for i in range(10):
        for j in range(10):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp  
    return list_index

def find_max(x, y):
    f = InterpolatedUnivariateSpline(x, y, k=4)
    cr_pts = f.derivative().roots()
    cr_pts = np.append(cr_pts, (x[0], x[-1]))  # also check the endpoints of the interval
    cr_vals = f(cr_pts)
    max_index = np.argmax(cr_vals)
    print("ep value:", numbers, "Maximum value {} at {}".format(cr_vals[max_index], cr_pts[max_index]))
    plt.plot()
    return cr_pts[max_index]


base = model.evaluate(x_test, y_test_one_hot)[1]
print(base)

rg = [*range(0, x_test.shape[0],1)]
epsilons = [0.20, 0.25, 0.30]
percent = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

   

for numbers in epsilons:
    acc = []
    for elements in percent:
        # reset accuracy counter every interation 
        z = 0 
        for i in range(len(rg)):        
            image = x_test[i]
            image_label = y_test[i]        
            # CREATE ADVERARIAL IMAGES
            generate_adverarial(image, image_label, numbers)
            # APPLY PERCENTILE FILTER
            perc = ndimage.percentile_filter(adv_img.reshape(28,28,1), percentile = elements, size = 2)
            # BLUR PRED       
            predict(perc)    
            if list_index[0] == image_label:
                z = z + 1                     
        accuracy = z / 10000
        print("ep value:", numbers, "percent:", elements, 'acc:', accuracy, "diff:", base - accuracy)
        acc.append(accuracy)
    print("ep value:", numbers, "acc:", acc)
    
    
    ## Interpolate generated data points to find max (shouls be highest possible max)
    max_x = find_max(percent, acc)
    maxsig_acc = 0 
    for i in range(len(rg)):        
        image = x_test[i]
        image_label = y_test[i]          
        # CREATE ADVERSARIAL IMAGES
        generate_adverarial(image, image_label, numbers)
        # BLUR ADVERAIRAL IMAGE (using guassian blurring)
        percMAXperc = ndimage.percentile_filter(adv_img.reshape(28,28,1), percentile = max_x, size = 2)
        # PREDICTION      
        predict(percMAXperc)       
        if list_index[0] == image_label:
            maxsig_acc = maxsig_acc + 1                    
    maximum = maxsig_acc / 10000
    print("ep value:", numbers, "maximum accuracy:", maximum, "max_x:", max_x, "diff:", base - accuracy)

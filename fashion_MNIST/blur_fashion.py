# Libraries
from fashion_MNIST import x_test, y_test, y_test_one_hot,model
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
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




############
#apply blur#
############

# number of image in the test dataset 
rg = [*range(0, x_test.shape[0],1)]

#blur coefficent (sigma). Change the elements in list to use different sigma values
sig = [0, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.8, 2.2]

# epsilon values. Change to use different epsilon values to be genertaed. These adverarial images will be blurred to increase accuracy. 
epsilons = [0.20, 0.25, 0.30]


print("baseline accuracy:", model.evaluate(x_test, y_test_one_hot)[1])


for numbers in epsilons:
    acc = []
    for elements in sig:
        # reset accuracy counter every interation 
        z = 0 
        for i in range(len(rg)): 
            image = x_test[i]
            image_label = y_test[i]   
            #Create advesarial images
            generate_adverarial(numbers)
            # Blur adverarial image (using guassian blurring)
            blur = ndimage.gaussian_filter(adv_img.reshape(28,28,1), sigma = elements)
            ### BLUR PRED       
            predict(blur)
            if list_index[0] == image_label:
                z = z + 1              
        accuracy = z / 10000
        print("ep value:", numbers, "blur coefficent:", elements, 'acc:', accuracy)
        acc.append(accuracy)
    print("epsilon value:", numbers, "acc:", acc)

    ## interpolate accruacy function data point ot find hgihest possible max
    max_x = find_max(sig, acc)
    maxsig_acc = 0 
    for i in range(len(rg)):        
        image = x_test[i]
        image_label = y_test[i]          
        # CREATE ADVERSARIAL IMAGES
        generate_adverarial(image, image_label, numbers)
        # BLUR ADVERAIRAL IMAGE (using guassian blurring)
        blurMAXsig = ndimage.gaussian_filter(adv_img.reshape(28,28,1), sigma = max_x)
        # PREDICTION      
        predict(blurMAXsig)       
        if list_index[0] == image_label:
            maxsig_acc = maxsig_acc + 1                    
    maximum = maxsig_acc / 10000
    print("ep value:", numbers, "maximum accuracy:", maximum, "max_x:", max_x)



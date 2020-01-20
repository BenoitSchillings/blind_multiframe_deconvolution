import tensorflow as tf

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import math
from scipy.ndimage.filters import gaussian_filter

import time
import os
import cv2
from glob import glob

#----------------------------------------------------------

def show_array(array, name):
    bg = np.percentile(array, 0.01)
    array = array - bg
    array = array / np.percentile(array, 99.8)
    
    cv2.imshow(name, np.sqrt(array*1.0))
    cv2.waitKey(1)

#----------------------------------------------------------
 
def show_array_linear(array, name):
    bg = np.percentile(array, 0.01)
    array = array - bg
    array = array / np.percentile(array, 99.8)
    
    cv2.imshow(name, array*1.0)
    cv2.waitKey(3)
 
#----------------------------------------------------------
   

def mask(image, size):
  blure = gaussian_filter(image, size)
  return (image - blure * 0.0)

#----------------------------------------------------------

ICOUNT = 30
PSF2 = 15
PSF = 31

def get_images():
    path_images =  glob('./data1/*.npy', recursive=True)
    count = len(path_images)
    count = ICOUNT
    
    model = np.load(path_images[0])
    show_array_linear(model, "ref")
    ims = np.zeros((count, model.shape[0], model.shape[1]))
   
    model = model - np.min(model)
    model = model / np.max(model)
    
    target_mean = np.median(model)
    sum = np.zeros((model.shape[0], model.shape[1]))
    for idx in range(count):
        an_image = np.load(path_images[idx])
        an_image = an_image - np.min(an_image)
        mean = np.mean(an_image)
        print(mean)
        
        an_image = an_image / (mean/target_mean)
        an_image = an_image + 1.0*np.random.standard_normal(model.shape)/152.0
        ims[idx] = an_image
        
        sum = sum + an_image
        show_array_linear(an_image, "tmp")
    
    sum = sum / count
    
    return ims, sum
    
#----------------------------------------------------------


def make_kernels(count):
    k = np.zeros((count, PSF, PSF))
    
    for i in range(count):
        k[i][PSF2][PSF2] = 1.0
        k[i] = gaussian_filter(k[i], 2.2)
        k[i] = k[i] + 0.0001
        k[i] = k[i] / np.sum(k[i])
        
    return k


images,sum = get_images()
kernels = make_kernels(images.shape[0])
print("kernels", kernels.shape)
  
#----------------------------------------------------------

#subset of the image we are going to play with


observed = tf.constant(images, name = 'observed', dtype=tf.float32)
psfs = tf.Variable(kernels, name = 'psf', dtype=tf.float32)
model = tf.Variable(sum, name = 'model', dtype=tf.float32)

total_error = 0.0

ADD = 4
for i in range(ICOUNT):
    psf = tf.square(psfs[i])
    result = tf.nn.conv2d(model[tf.newaxis, :, :, tf.newaxis],psf[:, :, tf.newaxis, tf.newaxis],strides=[1, 1, 1, 1],padding="VALID")[0, :, :, 0]
    norm = tf.reduce_sum(result)
    norm0 = tf.reduce_sum(observed[i][PSF2:-PSF2, PSF2:-PSF2])
    result = result * (norm0/norm)
    error = result[ADD:-ADD,ADD:-ADD] - observed[i][PSF2+ADD:-PSF2-ADD, PSF2+ADD:-PSF2-ADD]
    error = error * error
    #error = tf.abs(error)
    error_val = tf.reduce_sum(error)
    total_error = total_error + error_val

loss = tf.sqrt(total_error)

#----------------------------------------------------------

optimizer0 = tf.train.AdamOptimizer(0.0098)
optimizer1 = tf.train.AdamOptimizer(0.0007*10.0)
optimizer2 = tf.train.AdamOptimizer(0.0002)
train_model = optimizer0.minimize(loss, var_list=[model, psfs])
train_psf = optimizer1.minimize(loss, var_list=[psfs, model])
train_psf0 = optimizer2.minimize(loss, var_list=[psfs, model])

#----------------------------------------------------------

def optimize():
    with tf.Session() as session:
        init = tf.global_variables_initializer()
        session.run(init)    
        #for m in range(20):
        #        session.run(train_model)
        #for m in range(10):
        #        session.run(train_psf0)
        for k in range(100011):
            for m in range(15):
                session.run(train_model)
            for m in range(3):
                session.run(train_psf)
            #session.run(train_psf)
            res = session.run(result)
            show_array_linear(res, "result")
            res = session.run(psfs)
            #res = np.abs(res)
            res = np.square(res)
            show_array(res[0], "psf0")
            show_array(res[1], "psf1")
            show_array(res[2], "psf2")
            show_array(res[3], "psf3")
            print("psf = ", np.max(res))
            res = session.run(model)
            show_array_linear(res[20:-20,20:-20], "model")
            
            print("fit error = ",session.run(loss))


x = 0
optimize()

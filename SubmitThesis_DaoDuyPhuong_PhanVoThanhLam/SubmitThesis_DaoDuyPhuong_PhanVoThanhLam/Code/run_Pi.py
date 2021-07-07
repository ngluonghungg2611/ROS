import keras
import cv2
import numpy as np

import time
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout, Flatten
#from keras.layers import Conv2D, maxPooling2D
#from keras. import backend as K
from keras.utils import np_utils
from Config import *
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

cap= cv2.VideoCapture(0)
#model = load_model('model_Large_line_88.h5')
model = load_model('Model_fold_0_Epoch_0001.h5')

img_crop = []
try:
    
    while(True):
        #start = time.time()
        ret, img = cap.read()

        #img_crop = img[130:479,0:639]
        img_crop = cv2.resize(img, (200,66),interpolation=cv2.INTER_AREA)
        #cv2.imshow("anh", img_crop)        
        img_crop = img_crop.reshape(1,66,200,3)
        img_crop = img_crop.astype('float32')

        img_class = model.predict(img_crop)
        if img_class.argmax()== 0:
            control.control_steer(100)
            print("steer = 100");
        if img_class.argmax()== 1:
            control.control_steer(110)
            print("steer = 110");
        if img_class.argmax()== 2:
            control.control_steer(120)
            print("steer = 120");    
        if img_class.argmax()== 3:
            control.control_steer(130)
            print("steer = 130"); 
        if img_class.argmax()== 4:
            control.control_steer(140)
            print("steer = 140"); 
        if img_class.argmax()== 5:
            control.control_steer(150)
            print("steer = 150");   
        if img_class.argmax()== 6:
            control.control_steer(160) 
            print("steer = 160");                                                                                                          
        if cv2.waitKey(11) & 0xFF == 27:
            break
        
        #end = time.time()
        #seconds = end - start
        ## Calculate frames per second
        #fps  = 1 / seconds;
        #print("Estimated frames per second : {0}".format(fps))          
    cap.release()
    cv2.destroyAllWindows()

except KeyboardInterrupt:
   GPIO.cleanup()

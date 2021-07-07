import cv2
import os
import time
import serial
import numpy as np
import glob
from serial import Serial
port = serial.Serial(
    port='/dev/ttyUSB0',
    baudrate = 57600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)

number_img100 = 0
number_img110 = 0
number_img120 = 0
number_img130 = 0
number_img140 = 0
number_img150 = 0
number_img160 = 0

cap = cv2.VideoCapture(0)path100 = '/home/pi/Desktop/CODE/Image100'
path110 = '/home/pi/Desktop/CODE/Image110'
path120 = '/home/pi/Desktop/CODE/Image120'
path130 = '/home/pi/Desktop/CODE/Image130'
path140 = '/home/pi/Desktop/CODE/Image140'
path150 = '/home/pi/Desktop/CODE/Image150'
path160 = '/home/pi/Desktop/CODE/Image160'


while(True):
    if port.inWaiting()>0:
        x = port.readline()
        steer = x.decode("utf-8")[0:3]
        ret,frame = cap.read()
        #cv2.imshow('anh',frame)      
        if steer == '100':
            #frame = cv2.resize(frame, (200,66), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path100,'imagea%04d.jpg'%number_img100),frame)
            number_img100 += 1
        if steer == '110':
            #frame = cv2.resize(frame, (200,66), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path110,'imageb%04d.jpg'%number_img110),frame)
            number_img110 += 1
        if steer == '120':
            #frame = cv2.resize(frame, (200,66), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path120,'imagec%04d.jpg'%number_img120),frame)
            number_img120 += 1
        if steer == '130':
            #frame = cv2.resize(frame, (200,66), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path130,'imaged%04d.jpg'%number_img130),frame)
            number_img130 += 1
        if steer == '140':
            #frame = cv2.resize(frame, (200,66), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path140,'imagee%04d.jpg'%number_img140),frame)
            number_img140 += 1
        if steer == '150':
            #frame = cv2.resize(frame, (200,66), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path150,'imagef%04d.jpg'%number_img150),frame)
            number_img150 += 1
        if steer == '160':
            #frame = cv2.resize(frame, (200,66), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path160,'imageg%04d.jpg'%number_img160),frame)
            number_img160 += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





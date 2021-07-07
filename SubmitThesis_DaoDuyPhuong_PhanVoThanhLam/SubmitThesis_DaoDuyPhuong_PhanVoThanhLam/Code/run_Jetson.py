
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import serial

ser = serial.Serial(
	port = '/dev/ttyUSB0',
	baudrate = 115200,
	parity = serial.PARITY_NONE,
	stopbits = serial.STOPBITS_ONE,
	bytesize = serial.EIGHTBITS,
	timeout = 1)
	
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

from tensorflow.keras.models import load_model

model = load_model('0050.h5')

MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

NUM_CLASSES = 4

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
print("image_tensor :   {}".format(image_tensor))

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
print("detection_boxes :   {}".format(detection_boxes))

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
print("detection_scores :   {}".format(detection_scores))
print("detection_classes :   {}".format(detection_classes))

num_detections = detection_graph.get_tensor_by_name('num_detections:0')
print("num_detections :   {}".format(num_detections))

#--------------------PREDICT------------------------------------------
vid = cv2.VideoCapture(0)
starttime1 = 0
starttime2 = 0
starttime3 = 0
starttime4 = 0

time_delay = 1
number_img = 0
while True:
	ret, frame = vid.read()	
	#frame = frame[80:479,0:639]
	frame =cv2.resize(frame,(320,200))
	#frame = cv2.flip(frame,1)
	image_expanded = np.expand_dims(frame, axis=0)

	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: image_expanded})  
	if len(boxes[0]) >= 1 and scores[0][0] > 0.3:
		index = 0
		#print(boxes[0][0])  ###  y1 , x1, y2, x2 = boxes[0][0]
		cv2.rectangle(frame, (int(boxes[0][0][1]*320), int(boxes[0][0][0]*200)), (int(boxes[0][0][3]*320), int(boxes[0][0][2]*200)), (0, 255, 0), 2)

		if classes[0][index] == 1:
			if time.time() - starttime1 < time_delay:
				cv2.putText(frame, "Car", (int(boxes[0][0][1]*320), int(boxes[0][0][0]*200)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)
				##print(boxes[0][0])
			else: 
				cv2.putText(frame, "Car", (int(boxes[0][0][1]*320), int(boxes[0][0][0]*200)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)
				if max( int(boxes[0][0][2]*200) - int(boxes[0][0][0]*200), int(boxes[0][0][3]*320)-int(boxes[0][0][1]*320) ) > 106:
					ser.write(b'1')
					ser.flush()
					starttime1 = time.time()
					print("Car")

		else:
			frame_crop = frame[(int(boxes[0][0][0]*200)):(int(boxes[0][0][2]*200)),(int(boxes[0][0][1]*320)):(int(boxes[0][0][3]*320))]
			frame_resize = cv2.resize(frame_crop,(64,64),interpolation = cv2.INTER_AREA)
			cv2.imshow('Predict', frame_resize)
			frame_resize = frame_resize.reshape(1,64,64,3)
			img_class = model.predict(frame_resize) 
			if img_class.argmax() == 0:
				if time.time() - starttime2 < time_delay:
					cv2.putText(frame, "Left", (int(boxes[0][0][1]*320), int(boxes[0][0][0]*200)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)
				else:
					cv2.putText(frame, "Left", (int(boxes[0][0][1]*320), int(boxes[0][0][0]*200)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)
					if max( int(boxes[0][0][2]*200) - int(boxes[0][0][0]*200), int(boxes[0][0][3]*320)-int(boxes[0][0][1]*320) ) > 106:
						ser.write(b'3')
						ser.flush()
						starttime2 = time.time()
						print("Left")
			if img_class.argmax() == 1:
				if time.time() - starttime3 < time_delay:
					cv2.putText(frame,"Right", (int(boxes[0][0][1]*320), int(boxes[0][0][0]*200)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)
				else:
					cv2.putText(frame,"Right", (int(boxes[0][0][1]*320), int(boxes[0][0][0]*200)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)
					if max( int(boxes[0][0][2]*200) - int(boxes[0][0][0]*200), int(boxes[0][0][3]*320)-int(boxes[0][0][1]*320) ) > 106:
						ser.write(b'a')
						ser.flush()
						starttime3 = time.time()
						print("Right")
			if img_class.argmax() == 2:
				if time.time() - starttime4 < time_delay:
					cv2.putText(frame, "Stop", (int(boxes[0][0][1]*320), int(boxes[0][0][0]*200)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)
				else:
					cv2.putText(frame, "Stop", (int(boxes[0][0][1]*320), int(boxes[0][0][0]*200)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)
					if max( int(boxes[0][0][2]*200) - int(boxes[0][0][0]*200), int(boxes[0][0][3]*320)-int(boxes[0][0][1]*320) ) > 106:
						ser.write(b'b')
						ser.flush()
						starttime4 = time.time()
						print("Stop")		
	cv2.imshow('Object detector', frame)
	if cv2.waitKey(1) == 27:
		break
cv2.destroyAllWindows()


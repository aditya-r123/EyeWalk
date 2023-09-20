#importing all packages, setting up GPIO pins
import cv2
import os
import threading
import time
import RPi.GPIO as GPIO
os.environ['SDL_AUTODIRVER'] = 'dsp'
GPIO.setwarnings (False) 
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN) 

#assign GPIO pins to ultrasonic sensor pins
TRIG = 23 
ECHO = 24


#defining file paths for the code 
classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names" 
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")
#run code on pretrained model & files
configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

#defining machine learning model 
net = cv2.dnn_DetectionModel(weightsPath,configPath) #assign the paths to be studied by the model 
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#defining the function that calculates and reports the distance between EyeWalk and an object
def findDistance():
	try:
		while True:
			GPIO.output(TRIG, False)
			while GPIO.input(ECHO)==0: #while the ultrasound wave is outside of the sensor
				pulse_start = time.time() #time that the ultrasound is released 
			while GPIO.input(ECHO)==1: #while the ultrasound wave is not outside of the sensor

				pulse_end = time.time() #time that the ultrasound is returns to the ultrasonic sensor

			pulse_duration = pulse_end - pulse_start
			distance = pulse_duration = pulse_end - pulse_start
			distance = round(distance)
			os.system('espeak "{}"'.format(str(distance) + "centimeters away"))
			return
	except KeyBoardInterrupt

		GPIO.cleanup()

#defining the function that detects objects, runs the findDistance() method, and reports the object identified
def getObjects(img, thres, nms, draw=True, objects=[]): 
classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms) 
if len(objects) == 0: objects = classNames
objectInfo = []
if len(classIds) != 0:
for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):  
className = classNames[classId - 1]
if className in objects: 

os.system('espeak "{}"'.format(className + "detected"))
findDistance()
time.sleep(1)
objectInfo.append([box,className])
if (draw): 
cv2.rectangle(img,box,color=(0,255,0),thickness=2) 
cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) 
cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
		
return img,objectInfo


#similar to main(); assigns all the initial variables and runs the program constantly using a while loop
if __name__ == "__main__":
 
cap = cv2.VideoCapture(0) 
cap.set(3,640) 
cap.set(4,480) 

while True: success, img = cap.read() 
result, objectInfo = getObjects(img,0.55,0.2) 
cv2.imshow("Output",img) 
cv2.waitKey(1)

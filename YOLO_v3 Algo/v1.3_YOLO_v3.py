# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 13:48:59 2020

@author: Soham
"""


######################### importing libs #########################

import cv2
import numpy as np

######################### defining function #########################


#########################  taking inputs from mouse  clicks  #########################
    
image_coord_slot = cv2.imread('ref1.png')

# creating list_inside list for storing coord for each rows

print('how many rows')
row_num = int(input())
print('num of rows =', row_num)

L = []
for i in range(row_num ):
    L.append([]) # using f' string to represent each list
    
# generating the lists for each rows        
# function to display the coordinates of 
# of the points clicked on the image  
        
# list t store x coord of parking slots

for i in range(len(L)):
    
# function taking input from mouse click
    def click_event(event, x, y, flags, params): 
        # checking for left mouse clicks 
        if event == cv2.EVENT_LBUTTONDOWN: 
            
            L[i].append(x)
             
            # displaying the coordinates on the image
            font = cv2.FONT_HERSHEY_SIMPLEX 
            cv2.putText(image_coord_slot, str(x) , (x,y), font, 
                              0.5, (255, 0, 0), 2) 
            cv2.imshow('image', image_coord_slot) 
                 
                 
           # driver function 
    if __name__=="__main__": 
        # displaying the image 
        cv2.imshow('image', image_coord_slot) 
        
        # setting mouse hadler for the image 
        # and calling the click_event() function 
        cv2.setMouseCallback('image', click_event) 
    
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        
        
        
#################################   detecting slots   ####################################



def slot_detect(image1, list_row, y_lowerbound, y_upperbound):
    #list_row = coord containing list for each row
    # y_lowerbound = lower horizontal line for parking slot boundary of cars bumper facing towards you = 250
    # y_upperbound = upper horizontal line for parking slot boundary of cars rare back facing away from you= 220
    
    
    
    L_coord = []         # list for storing cooredinates of each parking slot from plot
    for i in range(0 , len(list_row)-1):
        x1 = list_row[i]
        x2 = list_row[i+1]
        L_coord.append((x1,x2))
        

##################################################################
                         #  YOLO  #
##################################################################
        
    image1 = cv2.resize(image1, None, fx=1, fy=1)
    height, width, channels = image1.shape
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            
        blob = cv2.dnn.blobFromImage(image1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
            
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        L_mean_box = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                        
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                if x >= 5 and (x+w) <= image1.shape[1] and (y+h) >= y_upperbound and (y+h)<= y_lowerbound :
                        
                    cv2.rectangle(image1, (x, y), (x + w, y + h), (0,0,255), 2)
                    mean_box = (x + x+w)/2     # avg of bounding box for cars
                    L_mean_box.append(mean_box)

        L_sorted = sorted(L_mean_box) # avg of x coord of boxes with cars
        print('mean of booked coord', L_sorted)
    

        L_sorted=L_sorted+[0]*(len(L_coord)-len(L_sorted))    # because the coord list is >= car box list    
        for i in L_coord:
            for j in L_sorted:
                if (j > i[0] and j <i[1]):
                    print("booked",L_coord.index(i)+1)
                    L_sorted=L_sorted[1:]
                    break
                else:
                    print("empty",L_coord.index(i)+1)
                    cv2.rectangle(image1, (i[0], y_upperbound - 20), (i[1], y_lowerbound -20 ), (0,255,0), 2)
                    cv2.putText(image1, "vacant", (i[0], y_lowerbound ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
                    continue   
    
    return(image1)
    

    
    
########################################## test results  ##########################################

vid = cv2.VideoCapture('testvideo.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
count = 0
success = True
fps = int(vid.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('RESULTS.avi', fourcc, 10, (750, 421)) 

  
while(vid.isOpened()):
    ret, frame=vid.read()
    image_read = frame
    if count%(5*fps) == 0 :
        
        image1 = cv2.resize(image_read, (750, 421))
        image1_1 = slot_detect(image1=image1,list_row = L[0], y_lowerbound=180, y_upperbound=160)
        image1_2 = slot_detect(image1=image1,list_row = L[1], y_lowerbound=250, y_upperbound=220)
        image1_3 = slot_detect(image1=image1,list_row = L[2], y_lowerbound=400, y_upperbound=350)
        superimposed_img1 = cv2.addWeighted(image1_1, 0.6, image1_2,0.6,1)
        superimposed_img1 = cv2.addWeighted(image1_3, 0.6, superimposed_img1, 0.7, 1)



        #out.write(superimposed_img1)
        cv2.imshow("Result", superimposed_img1)
    count+=1
    if cv2.waitKey(1)== ord('b'):
        break
vid.release()
#out.release()
cv2.destroyAllWindows()
                
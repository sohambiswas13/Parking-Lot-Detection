
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:43:38 2021

@author: Soham
"""


from skimage.measure import compare_ssim
import cv2
import numpy as np



######################### defining function #########################


#########################  taking inputs from mouse  clicks  #########################
    
# reference image
image_coord_slot = cv2.imread('ref_gif.png')
image_coord_slot = cv2.resize(image_coord_slot, (750,421))
################################ clicking event ################################
L = []

# function taking input from mouse click
def click_event(event, x, y, flags, params): 
    
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        
        
        L.append((y , x))
        
    # displaying the coordinates on the image
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.rectangle(image_coord_slot, (x-10, y), (x + 10, y + 20), (255,0,0), 2)
        cv2.rectangle(image_coord_slot, (x-10, y), (x + 60, y - 20 ), (255,0,0), -1)

        cv2.putText(image_coord_slot, 'Slot'+str(len(L)) , (x-2,y-4), font, 
                          0.5, (255, 255, 255), 1) 
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
        
#cv2.imshow('image', image_coord_slot) 
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 
       
###########################################


#before = cv2.imread('ref_gif.png')
#before = cv2.resize(before, (750,421))
#before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)

def detect_diff(before_gray,after_gray, L_coord):
    
        
    # Compute SSIM between two images
    (score, diff) = compare_ssim(before_gray, after_gray, full=True)
    print("Image similarity", score)
    
    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    
    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        print(area)
        if area > 500:
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(mask, [c], 0, (0,255,0), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
    #cv2.imshow('before', before)
    #cv2.imshow('after', after)
    #cv2.imshow('diff',diff)
    #cv2.imshow('mask',mask)
    #cv2.imshow('filled after',filled_after)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
            
    for i in L_coord :
        print(mask[i])
        if mask[i][1] == 255 :
            print('booked',i)
            cv2.rectangle(after, (i[1]-10,i[0]), (i[1]+10, i[0]-20), (0,0,255), 2)
            cv2.rectangle(after, (i[1]-10,i[0]), (i[1]+90, i[0]+20), (0,0,255), -1)
            cv2.putText(after, str(L.index(i)+1)+" booked", (i[1]-10,i[0]+13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255),1)
        else:
            print('vacant',i)       
            cv2.rectangle(after, (i[1]-10,i[0]), (i[1]+10, i[0]-20), (0,255,0), 2)
            cv2.rectangle(after, (i[1]-10,i[0]), (i[1]+90, i[0]+20), (0,255,0), -1)
            cv2.putText(after, str(L.index(i)+1)+" vacant", (i[1]-10,i[0]+13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0),1)
            


########################################## test results  ##########################################

vid = cv2.VideoCapture('testvideo3.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID') 

success = True
fps = int(vid.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('RESULTS.avi', fourcc, 10, (750, 421)) 

ret, frame=vid.read()
before = frame
before = cv2.resize(before, (750,421))
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)

while(vid.isOpened()):
    ret, frame=vid.read()
    

    after = frame
    after = cv2.resize(after, (750,421))
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
            
    detect_diff(before_gray,after_gray, L_coord=L)


    #out.write(after)
    cv2.imshow("Result", after)

    if cv2.waitKey(1)== ord('b'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()
                
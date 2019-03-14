# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 18:16:34 2019

@author: SzuHaoWu
"""
import cv2
import numpy as np

def isPointinPolygon(point, rangelist):  
    Xlist = []
    Ylist = []
    Px = point[0]
    Py = point[1]
    for i in range(len(rangelist)):
        Xlist.append(rangelist[i][0]) 
        Ylist.append(rangelist[i][1]) 
    maxX = max(Xlist)
    minX = min(Xlist)
    maxY = max(Ylist)
    minY = min(Ylist)    
    if (Px > maxX or Px < minX or 
        Py > maxY or Py < minY):  
        return False
    count = 0
    point1 = rangelist[0] 
    for i in range(1, len(rangelist)):    
        point2 = rangelist[i]
        # If the point is on any vertex of polygon.
        if (Px == point1[0] and Py == point1[1]) or (Px == point2[0] and Py == point2[1]):
            return False
        # If the intersection is on any vertex of polygon, I don't count it.
        elif ( Py == min ( point1[1],point2[1] )):
            point1 = point2
            continue
        # Distinguish if the ray may pass through the segments of polygon 
        if (point1[1] < Py and point2[1] >= Py) or (point1[1] >= Py and point2[1] < Py):
            x = point2[0] - (point2[1] - Py) * (point2[0] - point1[0])/(point2[1] - point1[1])
            if (x < Px): 
                count +=1
        # if the point is on any side of polygon
            elif (x == Px):
                return False                      
        point1 = point2
        
    if count%2 == 0:
        return False
    else:
        return True


boxes=[]
iter = 3
kernel = np.ones((5,5),np.uint8)
img = cv2.imread('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/trainset/18.png')
result_mask = np.load('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/result_mask.npy')
result_mask = result_mask.astype(np.uint8)
result_mask = cv2.erode(result_mask,kernel)
result_mask = cv2.dilate(result_mask, kernel, iterations = iter)
image, contours, hier = cv2.findContours(result_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
        if w/h>0.33 and w/h<0.714:
            boxes.append([x,y,x+w,y+h])
        elif w/h>1.4 and w/h<3:   
            boxes.append([x,y,x+w,y+h])
        

for i in range(1,len(boxes)-1):
    poly = [[boxes[i-1][0],boxes[i-1][1]],[boxes[i-1][0],boxes[i-1][3]],[boxes[i-1][2],boxes[i-1][3]],[boxes[i-1][2],boxes[i-1][1]]]
    if isPointinPolygon([boxes[i][0],boxes[i][1]],poly) or isPointinPolygon([boxes[i][0],boxes[i][3]], poly) or isPointinPolygon([boxes[i][2],boxes[i][3]],poly) or isPointinPolygon([boxes[i][2],boxes[i][1]],poly):
        boxes[i-1][0] = min(boxes[i-1][0],boxes[i][0])
        boxes[i-1][1] = min(boxes[i-1][1],boxes[i][1])
        boxes[i-1][2] = max(boxes[i-1][2],boxes[i][2])
        boxes[i-1][3] = max(boxes[i-1][3],boxes[i][3])
        boxes.pop(i)
        break
    poly = [[boxes[i+1][0],boxes[i+1][1]],[boxes[i+1][0],boxes[i+1][3]],[boxes[i+1][2],boxes[i+1][3]],[boxes[i+1][2],boxes[i+1][1]]]
    if isPointinPolygon([boxes[i][0],boxes[i][1]],poly) or isPointinPolygon([boxes[i][0],boxes[i][3]], poly) or isPointinPolygon([boxes[i][2],boxes[i][3]],poly) or isPointinPolygon([boxes[i][2],boxes[i][1]],poly):        
        boxes[i+1][0] = min(boxes[i+1][0],boxes[i][0])
        boxes[i+1][1] = min(boxes[i+1][1],boxes[i][1])
        boxes[i+1][2] = max(boxes[i+1][2],boxes[i][2])
        boxes[i+1][3] = max(boxes[i+1][3],boxes[i][3])
        boxes.pop(i)
        break
    
for i in range(len(boxes)):
    a = str(boxes[i][0])
    b = str(boxes[i][1])
    c = str(boxes[i][2])
    d = str(boxes[i][3])
    str1 = "["+a+","+b+"]"
    str2 = "["+c+","+d+"]"
    cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 0), 2)
    cv2.putText(img,str1,(boxes[i][0]-25,boxes[i][1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2)
    cv2.putText(img,str2,(boxes[i][2]-20,boxes[i][3]+15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2)

cv2.imshow("contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



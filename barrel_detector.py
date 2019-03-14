'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np
import math
from scipy.stats import multivariate_normal

class BarrelDetector():
    def __init__(self):
        '''
			Initilize your blue barrel detector with the attributes you need
			eg. parameters of your classifier
		'''
        
        self.meanbar = [127.237077407,72.8921150723,29.9765146813]
        self.meanblue = [127.034676999,89.921473838,58.5564905989]
        self.meannotblue = [88.0075681958,99.8430664613,107.606110685]
        self.meanother = [126.854981969,105.040470762,83.930350762]
        
        
        self.covbar = [[3140.39,0,0],
                       [0,1514.68,0],
                       [0,0,804.901]]
        self.covblue = [[2495.88,0,0],
                        [0,1707.8,0],
                        [0,0,1891.72]]
        self.covnotblue = [[3831.27,0,0],
                           [0,3592.43,0],
                           [0,0,3787.13]]
        self.covother = [[1923.6,0,0],
                        [0,1393.21,0],
                        [0,0,1487.6]]
       

        self.prior_blue = 0.25
        self.prior_notblue = 0.75
        self.prior_bar = 0.5
        self.prior_other = 0.5
    def segment_image(self, img):
        '''
			Calculate the segmented image using a classifier
			eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		# YOUR CODE HERE
        result_mask = np.zeros((800,1200))
        def pdf(x,mean,cov):

            (m,n,l) = x.shape
            out = np.zeros([m,n])
            cov_inv = np.linalg.pinv(cov)
            cov_det = np.linalg.det(cov)
            for i in range(m):
                for j in range(n):
                    diff = x[i][j]-mean
                    sc = 1/math.sqrt(2*math.pi*cov_det)
                    ex = np.exp((-0.5)*np.dot(np.dot(diff,cov_inv),(np.array([diff]).T)))
                    out[i][j] = sc*ex
            return out
          
        
        (m,n,l) = img.shape
        pro_blue = pdf(img,self.meanblue,self.covblue)
        pro_notblue = pdf(img,self.meannotblue,self.covnotblue)
        pro_bar = pdf(img,self.meanbar,self.covbar)
        pro_other = pdf(img,self.meanother,self.covother)

        conditionA = self.prior_blue*pro_blue > self.prior_notblue*pro_notblue
        conditionB = self.prior_bar*pro_bar > self.prior_other*pro_other 

        for i in range(m):
            for j in range(n):
                if conditionA[i][j] == True:
                    if conditionB[i][j] == True:
                        result_mask[i][j] = 1
                    else:
                        result_mask[i][j] = 0
                else:
                    result_mask[i][j] = 0
        return result_mask

    def get_bounding_box(self, img):
        '''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
		# YOUR CODE HERE
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
        result_mask = self.segment_image(img)
        result_mask = result_mask.astype(np.uint8)
        result_mask = cv2.erode(result_mask,kernel)
        result_mask = cv2.dilate(result_mask, kernel, iterations = iter)
        contours, hier = cv2.findContours(result_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
            cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 0), 2)
                
        return boxes


#if __name__ == '__main__':
#	folder = "trainset"
#	my_detector = BarrelDetector()
#	for filename in os.listdir(folder):
#		# read one test image
#		img = cv2.imread(os.path.join(folder,filename))
#		cv2.imshow('image', img)
#		cv2.waitKey(0)
#		cv2.destroyAllWindows()

		#Display results:
		#(1) Segmented images
		#	 mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box
		#    boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope


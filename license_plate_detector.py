import numpy as np
import cv2 
import math



def get_threshold(car_image):
    kernel = np.ones((5,5),'uint8')

    low_yellow = (10,80,108)
    high_yellow = (22,210,168)

    hsv_img = cv2.cvtColor(car_image,cv2.COLOR_BGR2HSV)

    threshsv = cv2.inRange(hsv_img,low_yellow,high_yellow)
    threshsv = cv2.blur(threshsv,(5,5))


    threshsv = cv2.morphologyEx(threshsv,cv2.MORPH_CLOSE,kernel,iterations=7)
    threshsv = cv2.blur(threshsv,(5,5))

    return threshsv


def is_good_ratio(box):# needs to work on rectangle orientation
    segment_1 = int(math.dist(box[0],box[3])) # width
    segment_2 = int(cv2.contourArea(box)/segment_1) # height
    
    
        
    retio_between_width_and_height = segment_1/segment_2
    
    return retio_between_width_and_height < 5.5 and retio_between_width_and_height > 2.5
    
     
    


def is_license_plate(contur):
    bounding_minimum_area_rectangle = cv2.minAreaRect(contur)
    box = cv2.boxPoints(bounding_minimum_area_rectangle)
    box = np.int0(box)
    x,y,w,h = cv2.boundingRect(contur)
    
    if cv2.contourArea(box) > 1000 and is_good_ratio(box) and (w*h)/cv2.contourArea(contur) < 1.7:
        return (x,y,w,h)



def main():
    image = cv2.imread('6.jpg')
    thresholded_car_image = get_threshold(image)

    conturs,_ = cv2.findContours(thresholded_car_image,cv2.RETR_EXTERNAL,cv2.RETR_CCOMP)
    
    #image = cv2.drawContours(image,conturs,-1,(0,255,0),3)
    
    
    license_plates = [is_license_plate(cont) for cont in conturs if is_license_plate(cont) != None]
    
    for (x,y,w,h),num in zip(license_plates,range(len(license_plates))):
        
        if w > h: #checking the orientation of the rectangle
            
            image[y:y+h,x:x+w] = [0,0,0]
       
       


    

    cv2.imshow('thresh',thresholded_car_image)

    cv2.imshow("car",image)


    cv2.waitKey(0)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
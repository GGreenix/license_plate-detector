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
    
    if segment_1 < segment_2:
        segment_1,segment_2 = segment_2,segment_1
        
    retio_between_width_and_height = segment_1/segment_2
    
    
    return retio_between_width_and_height < 6 and retio_between_width_and_height > 1.5
     
    


def First_license_plate_filter(contur):
    bounding_minimum_area_rectangle = cv2.minAreaRect(contur)
    box = cv2.boxPoints(bounding_minimum_area_rectangle)
    box = np.int0(box)
    x,y,w,h = cv2.boundingRect(contur)
    
    return cv2.contourArea(box) > 1000 and is_good_ratio(box) and (w*h)/cv2.contourArea(contur) < 1.7 and w > h
        
    
def get_minimum_bounding_box(contur):
    bounding_minimum_area_rectangle = cv2.minAreaRect(contur)
    box = cv2.boxPoints(bounding_minimum_area_rectangle)
    box = np.int0(box)
    x,y,w,h = cv2.boundingRect(contur)
    return x,y,w,h

def extract_license_from_image(car_image,thresh_image,parameters):  
    x,y,w,h = parameters
    mask = thresh_image[y:y+h,x:x+w]
    license_plate = car_image[y:y+h,x:x+w]
    license_plate = cv2.bitwise_and(license_plate,license_plate,mask=mask)
    return license_plate

#def check_for_average_black_in_license_plate  

def main():
    image = cv2.imread('6.jpg')
    thresholded_car_image = get_threshold(image)

    conturs,_ = cv2.findContours(thresholded_car_image,cv2.RETR_EXTERNAL,cv2.RETR_CCOMP)
    
    
    
    license_plates = [get_minimum_bounding_box(contur) for contur in conturs if First_license_plate_filter(contur)]
    
    
    
    for params,num in zip(license_plates,range(len(license_plates))):
        
        x,y,w,h = params
        
        cv2.imshow(str(num),extract_license_from_image(image,thresholded_car_image,params))
        image[y:y+h,x:x+w] = [0,0,0]
       
       


    

    cv2.imshow('thresh',thresholded_car_image)

    cv2.imshow("car",image)


    cv2.waitKey(0)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
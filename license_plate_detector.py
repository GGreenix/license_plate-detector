import numpy as np
import cv2 
import math



def get_threshold(car_image):

    '''
    input (original image)

    this fucntion is thresholding, blurring and generally proccesing the image for later use in the procces

    return (thresholded original image)
    '''
    kernel = np.ones((5,5),'uint8') #a 5x5 kernel for morphing the image and proccecing noise

    low_yellow = (10,80,108)
    high_yellow = (22,210,168)

    hsv_img = cv2.cvtColor(car_image,cv2.COLOR_BGR2HSV)

    thresholded_image = cv2.inRange(hsv_img,low_yellow,high_yellow)
    thresholded_image = cv2.blur(thresholded_image,(5,5))


    thresholded_image = cv2.morphologyEx(thresholded_image,cv2.MORPH_CLOSE,kernel,iterations=1)
    thresholded_image = cv2.blur(thresholded_image,(5,5))

    return thresholded_image


def is_good_ratio(box):
    '''
    input(minimum bounding rectagnle of given contur)

    this function is returning wether the given minimum ractangle of a given contur
    is between the ratio of the constant ratio of a license plate

    return( if given bounding rectangle ratio matches constant ratio of a license plate)

    '''

    segment_1 = int(math.dist(box[0],box[3])) # width
    segment_2 = int(cv2.contourArea(box)/segment_1) # height
    
    if segment_1 < segment_2:
        segment_1,segment_2 = segment_2,segment_1
        
    retio_between_width_and_height = segment_1/segment_2
    
    return inRange((2,6),retio_between_width_and_height)
    
     
    


def First_license_plate_filter(contur):# fix for it to be area and MINIMUM bounding rect area and not bigger bounding rect

    '''
    input (found contur coordinates numpy array)
    
    this function is taking the minimum bounding rectangle of a given
    contur and checks four different facotrs:
    - it's area is greater then 900 pixels
    - given minimum rectangle of given contur returns True from function is_good_area
    - if the ratio of the minimum bounding rectangle and the area is less then 1.7
    - width bigger then height

    return (if the given contur is all true in the given if statments below)
    '''
    bounding_minimum_area_rectangle = cv2.minAreaRect(contur)
    box = cv2.boxPoints(bounding_minimum_area_rectangle)
    box = np.int0(box)
    width_of_minimus_bounding_rect = math.dist((box[0][0],box[0][1]),(box[3][0],box[3][1]))
    height_of_minimus_bounding_rect = math.dist((box[0][0],box[0][1]),(box[1][0],box[1][1]))
    x,y,w,h = cv2.boundingRect(contur)
    
    return cv2.contourArea(box) > 900 and is_good_ratio(box) and (width_of_minimus_bounding_rect*height_of_minimus_bounding_rect)/cv2.contourArea(contur) < 1.7 and w > h
    
        
    
def get_minimum_bounding_box(contur):
    '''
    input (filtered contur (might be a license plate))

    this function is taking the given contur that
    might be a license plate after first filtering and 
    returning the bounding box of the minimum rectangle
    of the given contur

    return (bounding box of given contur)

    '''
    bounding_minimum_area_rectangle = cv2.minAreaRect(contur)
    box = cv2.boxPoints(bounding_minimum_area_rectangle)
    box = np.int0(box)
    x,y,w,h = cv2.boundingRect(contur)
    return x,y,w,h
    

def extract_license_from_image(license_plate_image,thresh_image,parameters):

    '''
    input (full image which contains the license plate, thresholded cars image, paramters of given license plate)

    this function is slicing a mask with the coordinates of the given license plate paramters
    from the thresholded image, and from the original image to much the sizes.
    then it is uses an and gate with the thresholded sliced mask as a mask and the original image
    to get the original license plate image.

    return (image with the size of the bounding rectangle which contains the license plate on x,y,w,h parameters)

    '''

    x,y,w,h = parameters
    mask = thresh_image[y:y+h,x:x+w]
    license_plate = license_plate_image[y:y+h,x:x+w]
    license_plate = cv2.bitwise_and(license_plate,license_plate,mask=mask)
    return license_plate


def inRange(range,value_to_compare):

    '''
    input(range(tuple), a numerical value)

    this function is checking if a given value is between a given range

    return(if the given value is between the given two values)
    '''

    return range[0] < value_to_compare and value_to_compare < range[1]


def is_pixel_black_enough(pixel):
    '''
    input (pixel value)

    this function is used for checking if given pixel HSV
    values are between constant range

    return (if H_value,S_value,V_value are between the constant range of (0,0,20)-(50,255,140))
    '''
    lower_black = (0,0,20)
    higher_black = (50,255,140)
    p_h,p_s,p_v = pixel[0],pixel[1],pixel[2]
    
    return inRange((lower_black[0],higher_black[0]),p_h) and inRange((lower_black[1],higher_black[1]),p_s) and inRange((lower_black[2],higher_black[2]),p_v)
    


def check_for_black_pixels_in_roi(image_roi,thresh_roi):
    '''
    input (region of image of most likely a license plate, region of image in the threshold image)
    
    this function is comparing side by side
    each and every pixel of the flattened region
    of image by, then it calculates the ratio
    between the aproved black pixels and the rest of the 
    thresholded pixels by summerizing all of the white pixels
    within the thresholded image

    return (if ratio of black pixels to the rest of the thresholded pixels is between 1-10)

    '''
    sum_of_black_pixels = 0 
    
    image_hsv_roi = cv2.cvtColor(image_roi,cv2.COLOR_BGR2HSV)
    new_shape = image_roi.shape[0]*image_roi.shape[1]                       #calculates new shape for reshaping the image for a better analasys
    thresh_roi = cv2.threshold(thresh_roi,1,255,cv2.THRESH_BINARY)[1]       #given thresholded image are being thresholded again because of blur and morphology 
    sum_of_thresholded_pixels = (thresh_roi == 255).sum()                   #summerize amount of white pixels in thresholded image
    flattened_hsv_img_roi,flattened_thresh_roi = image_hsv_roi.reshape(new_shape,3),thresh_roi.flatten()

    for i in flattened_hsv_img_roi:        
        if is_pixel_black_enough(i):
            sum_of_black_pixels += 1
    
    return inRange((1,10),sum_of_thresholded_pixels/sum_of_black_pixels)
    
            
        
    

def main():
    image = cv2.imread('9.jfif')
    
    thresholded_car_image = get_threshold(image)

    conturs,_ = cv2.findContours(thresholded_car_image,cv2.RETR_EXTERNAL,cv2.RETR_CCOMP)
    
    
    
    license_plates = [get_minimum_bounding_box(contur) for contur in conturs if First_license_plate_filter(contur)]
    
    
    for params in license_plates:
        
        x,y,w,h = params #parameters of the bounding rectangle
        
        
        if(check_for_black_pixels_in_roi(extract_license_from_image(image,thresholded_car_image,(x,y,w,h)),thresholded_car_image[y:y+h,x:x+w])):
            image[y:y+h,x:x+w] = [0,0,0] #blacks out all pixels within the region of the bounding rectangle
   
       


    

    cv2.imshow('thresh',thresholded_car_image)

    cv2.imshow("car",image)


    cv2.waitKey(0)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
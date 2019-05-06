# Main.py

import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import DetectChars
import DetectPlates
from PIL import Image
import PossiblePlate

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

def main(image):

    CnnClassifier = DetectChars.loadCNNClassifier()         # attempt KNN training
    response  = str(input('Do you want to see the Intermediate images: '))
    if response == 'Y' or response == 'y':
        showSteps = True
    else:
        showSteps = False


    if CnnClassifier == False:                               # if KNN training was not successful
        print("\nerror: CNN traning was not successful\n")               # show error message
        return                                                          # and exit program

    imgOriginalScene  = cv2.imread(image)               # open image
    
    h, w = imgOriginalScene.shape[:2]
    
    imgOriginalScene = cv2.resize(imgOriginalScene, (0, 0), fx = 1.4, fy = 1.4,interpolation=cv2.INTER_CUBIC)
    
    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")      # print error message to std out
        return                                              # and exit program

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates. We get a list of
                                                                                        # combinations of contours that may be a plate.


    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    if showSteps == True:
        Image.fromarray(imgOriginalScene,'RGB').show() # show scene image
        

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")             # inform user no plates were found
        response = ' '
        return response,imgOriginalScene
    else:                      
                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

        licPlate = listOfPossiblePlates[0]

        if showSteps == True:
            Image.fromarray(licPlate.imgPlate).show()    # show crop of plate and threshold of plate
            
        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            print("\nno characters were detected\n\n")       # show message
            return ' ',imgOriginalScene                                       # and exit program
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate

        print("\nlicense plate read from ", image," :",licPlate.strChars,"\n")
        print("----------------------------------------")

        if showSteps == True:
            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

            Image.fromarray(imgOriginalScene).show()                # re-show scene image

            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file
            input('Press any key to continue...')                    # hold windows open until user presses a key

    return licPlate.strChars,licPlate.imgPlate

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect. Here, bounding rectangle is drawn with minimum area, so it considers the rotation also

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function
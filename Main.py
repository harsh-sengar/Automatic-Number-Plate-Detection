import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import DetectChars
import DetectPlates
from PIL import Image
import PossiblePlate
import Main

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

def main(image):

    CnnClassifier = DetectChars.loadCNNClassifier()
    
    #print("yo")
    
    response  = str(input('Do you want to see the Intermediate images: '))

    if response == 'Y' or response == 'y':
        Main.showSteps = True
    else:
        Main.showSteps = False
    
    print(showSteps)
    
    if CnnClassifier == False:                              
        print("\nerror: CNN traning was not successful\n")              
        return                                                        

    imgOriginalScene  = cv2.imread(image)               
    plt.imshow(imgOriginalScene)
    h, w = imgOriginalScene.shape[:2]

    imgOriginalScene = cv2.resize(imgOriginalScene, (0, 0), fx = 1.4, fy = 1.4,interpolation=cv2.INTER_CUBIC)
    
    
    if imgOriginalScene is None:                           
        print("\nerror: image not read from file \n\n")
        return
    
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)         


    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates) 
    #print("listofPossiblePlate ")
    if showSteps == True:
        print("show kro1")
        Image.fromarray(imgOriginalScene,'RGB').show() 

    if len(listOfPossiblePlates) == 0:                          
        print("\nno license plates were detected\n")           
        response = ' '
        return response,imgOriginalScene
    else:
        # print("Else m aaya")
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

        licPlate = listOfPossiblePlates[0]

        if showSteps == True:
            print("ShowKaro2")
            Image.fromarray(licPlate.imgPlate).show()    
            
        if len(licPlate.strChars) == 0:                    
            print("\nno characters were detected\n\n")    
            return ' ',imgOriginalScene                                       
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             

        print("\nlicense plate read from ", image," :",licPlate.strChars,"\n")
        print("----------------------------------------")                

    return licPlate.strChars, licPlate.imgPlate

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)          

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)        
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
    

# if __name__ == "__main__":
   
#     main('OS269DT.jpg')

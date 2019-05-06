# DetectChars.py

import cv2
import numpy as np
import math
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
import Main
from PIL import Image
import Preprocess
import PossibleChar
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.optimizers import RMSprop
#from Main import showSteps
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 64
RESIZED_CHAR_IMAGE_HEIGHT = 64

MIN_CONTOUR_AREA = 100
model = load_model('New_model/char-reg.h5')

def loadCNNClassifier():
    model.compile(optimizer = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.005), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return True
def detectCharsInPlates(listOfPossiblePlates):
    #print("Main->showSteps", Main.showSteps)
    #print("detectCharsInPlates->showSteps", showSteps)
    #print("Main.Response->", Main.response)
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:         
        return listOfPossiblePlates            
  
    listOfPossiblePlates_refined = []
    for possiblePlate in listOfPossiblePlates:         
        
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)     # preprocess to get 
        
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6,interpolation=cv2.INTER_LINEAR)

              
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       
        if Main.showSteps == True:
            #print("showkaro3")
            Image.fromarray(possiblePlate.imgThresh).show()
            input('Press Enter to Continue....')
       
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if Main.showSteps == True:
            #print("showkaro4")
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]                                         # clear the contours list

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
           
            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
            
            imgContours = Image.fromarray(imgContours,'RGB')
            imgContours.show()
            input('Press Enter to Continue....')
              
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)
        if (len(listOfListsOfMatchingCharsInPlate) == 0):
            
            if Main.showSteps == True:
                #print("showkaro5")
                print("chars found in plate number " + str(intPlateCounter) + " = (none), press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1

            possiblePlate.strChars = ""
            continue                        

        if Main.showSteps == True:
            #print("showkaro6")
            #print("ListOfListOfMatchingChar")
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
             
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))

            imgContours = Image.fromarray(imgContours,'RGB')
            imgContours.show()
            input('Press Enter to Continue....')
       
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                             
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)       
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])   
            
        if Main.showSteps == True:
            imgContours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
              

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
         
            imgContours = Image.fromarray(imgContours,'RGB')
            imgContours.show()
            input('Press Enter to Continue....')
       
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
          
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        if Main.showSteps == True:
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            
            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
            imgContours = Image.fromarray(imgContours,'RGB')
            imgContours.show()
            
        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

        listOfPossiblePlates_refined.append(possiblePlate)

        if Main.showSteps == True:
            print("chars found in plate number " + str(intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
      
    if Main.showSteps == True:
        print("\nchar detection complete, press a key to continue . . .\n")
   
    return listOfPossiblePlates_refined 


def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                       
    contours = []
    imgThreshCopy = imgThresh.copy()

          
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                       
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):             
            listOfPossibleChars.append(possibleChar)    
  
    return listOfPossibleChars


def checkIfPossibleChar(possibleChar):

    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False

def findListOfListsOfMatchingChars(listOfPossibleChars):
          
    listOfListsOfMatchingChars = []                 
    for possibleChar in listOfPossibleChars:                      
        
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        
        listOfMatchingChars.append(possibleChar)                
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     
            continue
            
        listOfListsOfMatchingChars.append(listOfMatchingChars)
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)   
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:     
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)
        break;



    return listOfListsOfMatchingChars

def findListOfMatchingChars(possibleChar, listOfChars):
          
    listOfMatchingChars = []              

    for possibleMatchingChar in listOfChars:             
        if possibleMatchingChar == possibleChar:    
            continue                               
            
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

              
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)      
            
    return listOfMatchingChars               

def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))

def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                          
        fltAngleInRad = math.atan(fltOpp / fltAdj)     
    else:
        fltAngleInRad = 1.5708                        
    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)      

    return fltAngleInDeg

def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                
    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:       
                
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                              
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:        
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:             
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)       
                    else:                                                                      
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:               
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           

    return listOfMatchingCharsWithInnerCharRemoved



def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""           

    height, width = imgThresh.shape
    imgThreshColor = np.zeros((height, width, 3), np.uint8)
    thresholdValue, imgThresh = cv2.threshold(imgThresh, 0.0, 255.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  
    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)
 
    imgThreshColor2 = imgThreshColor.copy()
   
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)     

    for currentChar in listOfMatchingChars:                                       
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor2, pt1, pt2, (255,0,0), 2)         
        imgROI = imgThreshColor[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]
        imgROI = cv2.copyMakeBorder(imgROI,8,8,8,8,cv2.BORDER_CONSTANT,value = [255,255,255])

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT),interpolation=cv2.INTER_LINEAR)           
       
        img=np.reshape(imgROIResized,[1,64,64,3])
        
        if Main.showSteps == True:
            imgROIResized = cv2.resize(imgROIResized ,(250, 250))
            cv2.imshow("char", imgROIResized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        classes=model.predict_classes(img)
        if classes[0]<10:
            strCurrentChar = chr(classes[0]+48)
        else:
            strCurrentChar = chr(classes[0]+55)  
        if Main.showSteps == True:
        	print(strCurrentChar)
        strChars = strChars + strCurrentChar                       


    if Main.showSteps == True: 
        imgThreshColor2 = Image.fromarray(imgThreshColor2,'RGB')
        imgThreshColor2.show()
        input('Press Enter to Continue....')
  
    return strChars


# DetectPlates.py

import cv2
import numpy as np
import math
import Main
import random
import matplotlib.pyplot as plt
import Preprocess
import DetectChars
from PIL import Image
import PossiblePlate
import PossibleChar

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []                 
    #print("detectPlatesInScene->showSteps: ", Main.showSteps)
    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)
        
    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)
    
    if(Main.showSteps == True):
        imgcopy = imgThreshScene.copy()
        imgcopy = cv2.resize(imgcopy, (700,700))
        cv2.imshow('ThresholdImage', imgcopy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene) 
    

    if Main.showSteps == True: 
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(len(listOfPossibleCharsInScene)))

        imgContours = np.zeros((height, width, 3), np.uint8)
        contours = []
        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        imgOriginalcopy = imgOriginalScene.copy()
        imgOriginalcopy = cv2.resize(imgOriginalcopy, (700, 700))
        cv2.imshow('Image', imgOriginalcopy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #Image.fromarray(imgOriginalScene).show()
        input('Press any key to continue...')
            
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)
    
    if Main.showSteps == True: 
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(len(listOfListsOfMatchingCharsInScene)))    
        
        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
        
            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
    
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)         

        if possiblePlate.imgPlate is not None:                         
            listOfPossiblePlates.append(possiblePlate)                  
            

    if Main.showSteps == True:
        print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")
    if Main.showSteps == True: 
        print("\n")
        
        imgcopy = imgContours.copy()
        imgcopy = cv2.resize(imgcopy, (700, 700))
        cv2.imshow("Possible Plates", imgcopy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #Image.fromarray(imgContours,'RGB').show()
        input('Press any key to continue...')
        
        print("\nplate detection complete, press a key to begin char recognition . . .\n")
        input()
   
    return listOfPossiblePlates


def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []                # this will be the return value

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()
    #print('Now we start to find the contours in the thresholded image that may be characters:')

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # find contours

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                       # for each contour
            
        possibleChar = PossibleChar.PossibleChar(contours[i]) 
        if DetectChars.checkIfPossibleChar(possibleChar):                   
            intCountOfPossibleChars = intCountOfPossibleChars + 1           
            listOfPossibleChars.append(possibleChar)                        
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)
            
    if Main.showSteps == True:
        print("\nstep 2 - Total number of contours found in the image are = " + str(len(contours)))
        print("step 2 - number of contours those may be characters = " + str(intCountOfPossibleChars))
        imgcopy = imgContours.copy()
        imgcopy = cv2.resize(imgcopy, (520, 520))
        cv2.imshow("AllContours", imgcopy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #Image.fromarray(imgContours,'RGB').show()
    return listOfPossibleChars

def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()       
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)       
    
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0
    
    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

         
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR) 
    
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

         
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0) 

    height, width, numChannels = imgOriginal.shape     
    
    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))      

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped        

    return possiblePlate

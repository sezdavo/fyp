from pymongo import MongoClient
import numpy as np
import math
from tqdm import tqdm

# Function for turning array id values into strings for saving
def idValue2String(value):
    if value < 10:
        string = '0000' + str(value)
    elif value < 100:
        string = '000' + str(value)
    elif value < 1000:
        string = '00' + str(value)
    elif value < 10000:
        string = '0' + str(value)
    elif value < 100000:
        string = str(value)
    else:
        pass
    return string

# SETUP DATABASE
cluster = MongoClient("mongodb+srv://sezdavo:Sezmongo1012!@cluster0-wmv2v.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["PIE"]
collection = db["pedreps"]

# Grab data from mongo DB
results = collection.find({})

# Define ID for saving arrays as .npy files
arrayID = 1
# Define diagnostics counters for end evaluation
errorCount = 0
resultCount = 0

# LOOP THROUGH ALL PEDESTRIAN REPRESENTATIONS
for result in tqdm(results):
    resultCount += 1
    # Grab all info from representation
    pedID = result['_id']
    startFrame = result['startFrame']
    endFrame = result['endFrame']
    # print("The end frame is: " + endFrame)
    trackTime = result['trackTime']
    itemsList = result['representation']

    # Build frame index arrays for each 3 seconds encoded chunk of data desired
    # Define first set of start and end frames
    startFrame = int(startFrame)
    # Build arrays
    arrayArray = [[], [], []]
    indexFrame = int(startFrame)
    currentFrame = startFrame
    # Loop that creates n index arrays based on temporal resolution
    for array in arrayArray:
        while currentFrame < int(endFrame):
            array.append(currentFrame)
            currentFrame += 3
        indexFrame += 1
        currentFrame = indexFrame
    # print('These are the three long arrays:')
    # print(arrayArray)
    # Loop that splits these large arrays into chunks based on desired encoding time
    kingArray = []
    for array in arrayArray:
        arrayLength = len(array)
        for i in range(0, arrayLength, 30):
            x = array[i:i + 30]
            if len(x) == 30:
                kingArray.append(x)

    
    # LOOP THROUGH ALL INDEX ARRAYS TO BUILD DATA CHUNKS
    for array in kingArray:
        # print('These are the frames being extracted:')
        # print(array)
        # Empty chunk array    
        chunkArray = [] 
        # LOOP THROUGH ALL ITEMS AND APPEND DESIRED FRAME OBJECTS
        for item in itemsList:
            # Grab info needed for array
            frame = float(item['frame'])
            # Check if frame is part of array 
            match = 0
            for frame in array:
                if frame == int(item['frame']):
                    match = 1
            # If there is a match, extract info and append
            if match == 1:
                # Grab info needed for array
                frame = float(item['frame'])
                # Keep grabbing info
                xCentre = float(item['xCentre'])
                yCentre = float(item['yCentre'])
                boxArea = float(item['boxArea'])
                class1 = item['objectClass'][0]
                class2 = item['objectClass'][1]
                class3 = item['objectClass'][2]
                class4 = item['objectClass'][3]
                class5 = item['objectClass'][4]
                if item['objectID'] == pedID:
                    itself = 1
                    cross = item['cross']
                    action = item['action']
                    print([cross, action])
                else:
                    itself = 0
                # Build array
                objectArray = [frame, xCentre, yCentre, boxArea, class1, class2, class3, class4, class5, itself]
                # Turn into numpy array
                #objectArray = np.array(objectArray)
                chunkArray.append(objectArray)
            else:
                pass
        
        try:
            # Iterate through chunk array and find query object index
            i = 0
            index = 0
            for chunk in chunkArray:
                if chunk[4] == 1:
                    index = i
                i += 1

            # Move query object to the end of the sequence
            length = len(chunkArray)
            chunkArray.append(chunkArray.pop(index))
            prediction = [action, cross]
            package = [np.array(chunkArray), prediction]
            # Turn chunk array into numpy array
            package = np.array(package, dtype=object)
            # Save array
            saveString = idValue2String(arrayID)
            np.save('/Users/eliot/Documents/FYP/YoloV5/PIE/testdata/' + saveString + '.npy', package)
            # print("saved array " + saveString)
            # print("of length: " + str(len(chunkArray)))
            arrayID += 1
        except Exception as e:
            errorCount += 1
            # print(e)


print(errorCount)
print(resultCount)
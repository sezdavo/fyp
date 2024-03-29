#sImport necessart libraried
import math
from tqdm import tqdm
import pickle as pkl
import os
from os import walk

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
# Grab pickled files and append to results list
results = []
mypath = '/home/azureuser/cloudfiles/code/Users/esd27/pedreps'
_, _, filenames = next(walk(mypath))
# Loop through files and grab IF type = pickle file
for f in tqdm(filenames):
    if f.endswith(".p"):
        results.append(f)

# Define ID for saving arrays as .p files
arrayID = 1
# Define diagnostics counters for end evaluation
errorCount = 0
resultCount = 0
crossCount = 0
notCrossCount = 0
walkCount = 0
standCount = 0

# LOOP THROUGH ALL PEDESTRIAN REPRESENTATIONS
for result in tqdm(results):
    resultCount += 1
    # Load pickle file
    with open('/home/azureuser/cloudfiles/code/Users/esd27/pedreps/' + result, "rb") as f:
        pickle = pkl.load(f)
        # Grab all info from representation
        pedID = pickle['_id']
        startFrame = pickle['startFrame']
        endFrame = pickle['endFrame']
        # print("The end frame is: " + endFrame)
        trackTime = pickle['trackTime']
        itemsList = pickle['representation']
    # Get OBD array for speed values
    # Get set ID
    setID = '0' + pedID[0]
    # Get video ID by checking whether nuber is double or singled digit
    videoID = None
    try:
        int(pedID[2])
        videoID = pedID[2]
    except Exception as e:
        pass
    try:
        int(pedID[3])
        videoID = videoID + pedID[3]
    except Exception as e:
        pass
    if len(videoID) == 2:
        videoID = videoID
    else:
        videoID =  '0' + videoID
    # make string
    string = 'set' + setID + 'video' + videoID + '.p'
    # Load speed array for indexing
    with open('/home/azureuser/cloudfiles/code/Users/esd27/speedArrays/' + string, "rb") as f:
        speedArray = pkl.load(f)
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
        foundFrame = array[0]
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
                relativeFrame = frame - foundFrame
                relativeTime = relativeFrame / 30
                if relativeTime < 0:
                    print(relativeTime)
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
                else:
                    itself = 0
                # Get speed
                speed = speedArray[int(frame)]
                # Build array
                objectArray = [relativeTime, xCentre, yCentre, boxArea, itself, speed, class1, class2, class3, class4, class5, 0]
                # Append onto main array
                chunkArray.append(objectArray)
            else:
                pass
        
        # Use try statement to catch and print errors
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
            # EVAL
            if prediction[1] == 1:
                crossCount += 1
            elif prediction[1] == 0:
                notCrossCount += 1
            if prediction[0] == 1:
                walkCount += 1
            elif prediction[0] == 0:
                standCount += 1  
            # package = [np.array(chunkArray), prediction]
            package = [chunkArray, prediction]
            # print(chunkArray[length-1])
            # Turn chunk array into numpy array
            # package = np.array(package, dtype=object)
            # Save array
            saveString = idValue2String(arrayID)
            # np.save('piedata/' + saveString + '.npy', package)
            pkl.dump(package, open('/home/azureuser/cloudfiles/code/Users/esd27/piedatanew/' + saveString + '.p', 'wb'))
            # print("saved array " + saveString)
            # print("of length: " + str(len(chunkArray)))
            arrayID += 1
        except Exception as e:
            errorCount += 1
            print(e)
            # print(result)

# Print evaluation metrics
print('errorCount')
print(errorCount)
print('resultCount')
print(resultCount)
print('crossCount')
print(crossCount)
print('notCrossCount')
print(notCrossCount)
print('standCount')
print(standCount)
print('walkCount')
print(walkCount)
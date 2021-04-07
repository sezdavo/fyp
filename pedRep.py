from pymongo import MongoClient
from tqdm import tqdm
import pickle as pkl

# SETUP DATABASE
cluster = MongoClient("mongodb+srv://sezdavo:Sezmongo1012!@cluster0-wmv2v.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["PIE"]
collection = db["reps"]
collection2 = db["pedreps"]

# Grab data from mongo DB
results = collection.find({})

# function for getting appropriate representation between start and end frame for a pedestrian track
def sliceRepresentation(startFrame, endFrame, itemsList):
    newItemsList = []
    for item in itemsList:
        if int(item['frame']) >= int(startFrame) and int(item['frame']) <= int(endFrame):
            newItemsList.append(item)
    return newItemsList

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

# File saving name counter
counter = 0
idx = 0
for result in tqdm(results):
    # Bodged bug fix due to clip splitting in previous phase (rejoins clips)
    if len(result['_id']) == 16:
        itemsList = result['representation']
        idx += 1
    elif result['_id'][-1] == '1':
        rep1 = result['representation']
        string = result['_id'][:-1] + '2'
        nextOne = collection.find_one({'_id': string})
        rep2 = nextOne['representation']
        print('merged two clips')
        print(len(rep1))
        print(len(rep1))
        itemsList = rep1 + rep2
        print(len(itemsList))
        idx += 1
    elif result['_id'][-1] == '2':
        idx += 1
        continue
               
    # Grab representation from mongo item
    # itemsList = result['representation']
    # ITEM STRUCTURE:
    #miniDict = {
    #                'frame': frameNumber,
    #                'xCentre': str(annotationArray[0]),
    #                'yCentre': str(annotationArray[1]),
    #                'boxArea': str(annotationArray[2]*annotationArray[3]),
    #                'objectClass': vector,
    #                'action': action,
    #                'cross': cross,
    #                'objectID': objectID,
    #                'startFrame': startFrame,
    #                'endFrame': endFrame
    #            }

    # Loop through items to grab indexes for all pedestrians with a specific box area
    i = 0
    pedIndexes = []
    pedFrames = []
    for item in itemsList:  
        if item['objectClass'] == [1, 0, 0, 0, 0]:
            pedIndexes.append(i)
            pedFrames.append(item['frame'])
        else:
            pass
        i += 1

    # Loop through items and grab indexes of first instance of a new pedestrian track using ID
    knownIDs = []
    pedInstance = []

    n = 0
    for item in itemsList:  
        if item['objectClass'] == [1, 0, 0, 0, 0]:
            match = 0
            for knownID in knownIDs:
                if item['objectID'] == knownID:
                    match = 1
                    break
            if match == 0:
                knownIDs.append(item['objectID'])
                pedInstance.append(n)   
        else:
            pass
        n += 1
        # print(knownIDs)
        # print(pedInstance)

    # print(pedInstance)

    # Create database of individual pedestrians with their corresponding start and end frames
    pedestrianLocations = []
    for index in pedInstance:
        pedestrianID = itemsList[index]['objectID']
        # print(itemsList[index]['objectID'])
        startFrame = itemsList[index]['startFrame']
        endFrame = itemsList[index]['endFrame']
        seconds = (int(endFrame) - int(startFrame)) / 30 
        # Function that slices out relevant objects from clip representation
        slicedRepresentation = sliceRepresentation(startFrame, endFrame, itemsList)
        post = {
            '_id': pedestrianID,
            'startFrame': startFrame,
            'endFrame': endFrame, 
            'trackTime': seconds,
            'representation': slicedRepresentation
        }
        
        name = idValue2String(counter)
        pkl.dump(post, open('/Users/eliot/Documents/FYP/YoloV5/PIE/pedReps/' + name + '.p', 'wb'))
        counter += 1
        # collection2.insert_one(post)






# For each pedestrian create a new items list with new information
#dict = {
#                'pedestrianID': frameNumber,
#                'xCentre': str(annotationArray[0]),
#                'yCentre': str(annotationArray[1]),
#                'boxArea': str(annotationArray[2]*annotationArray[3]),
#                'objectClass': vector,
#                'action': action,
#                'cross': cross,
#                'objectID': objectID,
#                'startFrame': startFrame,
#                'endFrame': endFrame
#            }
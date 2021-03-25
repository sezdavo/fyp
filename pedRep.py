from pymongo import MongoClient

# SETUP DATABASE
cluster = MongoClient("mongodb+srv://sezdavo:Sezmongo1012!@cluster0-wmv2v.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["PIE"]
collection = db["reps"]
collection2 = db["pedreps"]

# Grab data from mongo DB
results = collection.find({})

# Grab representation from mongo item
itemsList = results[0]['representation']
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

# function for getting appropriate representation between start and end frame for a pedestrian track
def sliceRepresentation(startFrame, endFrame, itemsList):
    newItemsList = []
    for item in itemsList:
        if int(item['frame']) >= int(startFrame) and int(item['frame']) <= int(endFrame):
            newItemsList.append(item)
    return newItemsList

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
    print(knownIDs)
    print(pedInstance)

print(pedInstance)

# Create database of individual pedestrians with their corresponding start and end frames
pedestrianLocations = []
for index in pedInstance:
    pedestrianID = itemsList[index]['objectID']
    print(itemsList[index]['objectID'])
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
    collection2.insert_one(post)






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
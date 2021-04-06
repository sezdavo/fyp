import xml.etree.ElementTree as ET
from pymongo import MongoClient

# SETUP DATABASE
cluster = MongoClient(
    "mongodb+srv://sezdavo:Sezmongo1012!@cluster0-wmv2v.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["PIE"]
collection = db["reps"]


# Function for converting from PIE format to YOLO format
def pie2yolo(h, w, xbr, xtl, ybr, ytl):
    # CALCULATE PARAMETERS IN PIXELS FROM INPUT
    boxWidthPixels = xbr - xtl
    boxHeightPixels = ybr - ytl
    xCentrePixels = xtl + (boxWidthPixels/2)
    yCentrePixels = ytl + (boxHeightPixels/2)
    # CONVERT TO VALUE BETWEEN 0 AND 1 (PROPORTION)
    boxWidth = boxWidthPixels / w
    boxHeight = boxHeightPixels / h
    xCentre = xCentrePixels / w
    yCentre = yCentrePixels / h
    annotationArray = [xCentre, yCentre, boxWidth, boxHeight]
    return annotationArray

# Function for turning type of object into classification vector


def classifierVector(name):
    # Create classifier vector
    if name == 'pedestrian':
        vector = [1, 0, 0, 0, 0]
    elif name == 'vehicle':
        vector = [0, 1, 0, 0, 0]
    elif name == 'traffic_light':
        vector = [0, 0, 1, 0, 0]
    elif name == 'sign':
        vector = [0, 0, 0, 1, 0]
    elif name == 'crosswalk':
        vector = [0, 0, 0, 0, 1]
    else:
        vector = [0, 0, 0, 0, 0]
    return vector

# Function that converts action strings to numbers


def actionConvert(action):
    if action == 'walking':
        action = 1
    elif action == 'standing':
        action = 0
    else:
        pass
    return action

# Function that converts cross strings to numbers


def crossConvert(cross):
    if cross == 'crossing':
        cross = 1
    elif cross == 'not-crossing':
        cross = 0
    elif cross == 'crossing-irrelevant':
        cross = -1
    else:
        pass
    return cross


tree = ET.parse(
    '/Users/eliot/Documents/FYP/YoloV5/PIE/annotations/set04/video_0015_annt.xml')

root = tree.getroot()

# create empty list of dicts (each dict represents an object in a frame)
dictList = []
counter = 0

# Number of tracks in clip
print(len(tree.findall('track')))

# Loop through every object in clip
for track in tree.findall('track'):
    # Grab all frames associated with the object
    boxList = track.findall('box')
    # Grab length of boxList for grabbing start and end frames
    boxLength = len(boxList)
    startFrame = boxList[0].attrib.get('frame')
    endFrame = boxList[boxLength-1].attrib.get('frame')
    # Get object ID
    objectID = boxList[0].findall('attribute')[0].text
    # Loop through boxes and grab information
    for box in boxList:
        # Get frame number
        frameNumber = box.attrib.get('frame')
        # Get bounding box information
        xbr = float(box.attrib.get('xbr'))
        xtl = float(box.attrib.get('xtl'))
        ybr = float(box.attrib.get('ybr'))
        ytl = float(box.attrib.get('ytl'))
        annotationArray = pie2yolo(1080, 1920, xbr, xtl, ybr, ytl)
        # Get classifier vector
        vector = classifierVector(track.attrib.get('label'))
        # Get action
        if track.attrib.get('label') == 'pedestrian':
            action = box.findall('attribute')[3].text
            cross = box.findall('attribute')[4].text
        else:
            action = 'NaN'
            cross = 'NaN'
        # Convert actions and cross to 1 or 0 or -1
        action = actionConvert(action)
        cross = crossConvert(cross)

        # Construct mini dictionary
        miniDict = {
            'frame': frameNumber,
            'xCentre': str(annotationArray[0]),
            'yCentre': str(annotationArray[1]),
            'boxArea': str(annotationArray[2]*annotationArray[3]),
            'objectClass': vector,
            'action': action,
            'cross': cross,
            'objectID': objectID,
            'startFrame': startFrame,
            'endFrame': endFrame
        }
        # Check to see dictList contains any values

        # Define appended tracker
        appended = 0
        # Define index tracker
        index = 0
        if dictList:
            # Loop through all items in dictList
            for i in dictList:
                # Check if frame attribute is greater than frame attribute of dict being appended
                if int(i['frame']) > int(miniDict['frame']):
                    # insert dict infront of dict with higher frmae number
                    dictList.insert(index-1, miniDict)
                    appended = 1
                    break
                else:
                    pass
                index += 1
            if appended == 0:
                dictList.append(miniDict)
        else:
            dictList.append(miniDict)
        counter += 1

# f = open("representation.txt", "a")


# CREATE MONGO POST
post = {"_id": 'set04/video_0015',
        "representation": dictList
        }

collection.insert_one(post)

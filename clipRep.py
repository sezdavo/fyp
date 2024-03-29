import xml.etree.ElementTree as ET
from pymongo import MongoClient
from tqdm import tqdm

# SETUP DATABASE
cluster = MongoClient("mongodb+srv://sezdavo:Sezmongo1012!@cluster0-wmv2v.mongodb.net/test?retryWrites=true&w=majority")
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

def getStateValue(state, box):
    if state == 'action':
        # Get action attribute text
        for attribute in box.findall('attribute'):
            if attribute.attrib.get('name') == 'action':
                action = attribute.text
        # Convert
        if action == 'walking':
            action = 1
        elif action == 'standing':
            action = 0
        else:
            pass
        return action
    elif state == 'cross':
        # Get cross attribute text
        for attribute in box.findall('attribute'):
            if attribute.attrib.get('name') == 'cross':
                cross = attribute.text
        # Convert
        if cross == 'crossing':
            cross = 1
        elif cross == 'not-crossing':
            cross = 0
        elif cross == 'crossing-irrelevant':
            cross = -1
        else:
            pass
        return cross
    else:
        pass

# Function that converts action strings to numbers
# def actionConvert(action):
#     if action == 'walking':
#         action = 1
#     elif action == 'standing':
#         action = 0
#     else:
#         pass
#     return action

# Function that converts cross strings to numbers
# def crossConvert(cross):
#     if cross == 'crossing':
#         cross = 1
#     elif cross == 'not-crossing':
#         cross = 0
#     elif cross == 'crossing-irrelevant':
#         cross = -1
#     else:
#         pass
#     return cross

# SET FILE ID'S
set01Files = ['01', '02', '03', '04']
set02Files = ['01', '02', '03']
set03Files = ['01', '02', '03', '04', '05', '06', '07', '08',
              '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
set04Files = ['01', '02', '03', '04', '05', '06', '07',
              '08', '09', '10', '11', '12', '13', '14', '15', '16']
set05Files = ['01', '02']
set06Files = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
# SET LIST FOR LOOP
# setList = [set01Files, set02Files, set03Files,
#            set04Files, set05Files, set06Files]
setList = [set02Files, set03Files,
           set04Files, set05Files, set06Files]         

# SET ID'S FOR LOOP
# setIndex = ['01', '02', '03', '04', '05', '06']
setIndex = ['02', '03', '04', '05', '06']

# FOR LOOP TO LOOP THROUGH SETS (6 TOTAL)
for i in setList:

    setXFiles = i

    if i == set01Files:
        setID = '01'
    elif i == set02Files:
        setID = '02'
    elif i == set03Files:
        setID = '03'
    elif i == set04Files:
        setID = '04'
    elif i == set05Files:
        setID = '05'
    elif i == set06Files:
        setID = '06'
    else:
        pass

    sumArray = []

    for f in tqdm(setXFiles):

        tree = ET.parse('/Users/eliot/Documents/FYP/YoloV5/PIE/annotations/set' + setID + '/video_00' + f + '_annt.xml')
                
        root = tree.getroot()

        # create empty list of dicts (each dict represents an object in a frame)
        dictList = []
        counter = 0

        # Number of tracks in clip
        # print(len(tree.findall('track')))

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
                    action = getStateValue('action', box)
                    cross = getStateValue('cross', box)
                else:
                    action = 'NaN'
                    cross = 'NaN'
                # Convert actions and cross to 1 or 0 or -1
                # action = actionConvert(action)
                # cross = crossConvert(cross)

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

        # Split into two posts
        length = len(dictList)
        idx = int(length / 2)
        dictList1 = dictList[:idx]
        dictList2 = dictList[idx:]

        # CREATE MONGO POST
        post = {"_id": 'set' + setID + '/video_00' + f + '/01',
                "representation": dictList1
                }
        post2 = {"_id": 'set' + setID + '/video_00' + f + '/02',
                "representation": dictList2
                }

        collection.insert_one(post)
        collection.insert_one(post2)


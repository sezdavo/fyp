import cv2
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm

# framen number to string
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

# Grab one example frame from clip and annotate it
# For this we will use
repPath = '/Users/eliot/Documents/FYP/Implementation/clip.p'
# imagesPath = '/Users/eliot/Documents/FYP/YoloV5/PIE/images/set04/video_0015/'
imagesPath = '/Users/eliot/Documents/FYP/YoloV5/PIE/images/set04/video_0015/New Folder With Items/'
saveLocation = '/Users/eliot/Documents/FYP/YoloV5/PIE/images/set04/video_0015/New Folder With Items/'
# Now we need all of the annotations for this frame
# EACH OBJECT IN CLIP REP LOOKS LIKE THIS
# miniDict = {
#                 'frame': frameNumber,
#                 'xCentre': str(annotationArray[0]),
#                 'yCentre': str(annotationArray[1]),
#                 'boxArea': str(annotationArray[2]*annotationArray[3]),
#                 'boundingbox': [xbr, xtl, ybr, ytl],
#                 'objectClass': vector,
#                 'action': action,
#                 'cross': cross,
#                 'objectID': objectID,
#                 'startFrame': startFrame,
#                 'endFrame': endFrame
#             }
# Load clip representation
with open(repPath, "rb") as f:
        pickle = pkl.load(f)
# Load cross predictions
with open('/Users/eliot/Documents/FYP/Implementation/crossDict.p', "rb") as f:
        pickle2 = pkl.load(f)
# Load cross predictions
with open('/Users/eliot/Documents/FYP/Implementation/walkDict.p', "rb") as f:
        pickle3 = pkl.load(f)
# Loop through all items in clip
for item in tqdm(pickle):
    try:
        # Get name
        if item['objectClass'][0] == 1:
            name = 'pedestrian'
        elif item['objectClass'][1] == 1:
            name = 'vehicle'
        elif item['objectClass'][2] == 1:
            name = 'traffic light'
        elif item['objectClass'][3] == 1:
            name = 'sign'
        else:
            name = 'crosswalk'
        # Get bounding box info
        xbr = int(float(item['boundingbox'][0]))
        xtl = int(float(item['boundingbox'][1]))
        ybr = int(float(item['boundingbox'][2]))
        ytl = int(float(item['boundingbox'][3]))
        # Get frame
        frameNumber = item['frame']
        # Get image file name
        string = idValue2String(int(frameNumber))
        imageID = string + '.png'
        # Make image path
        imagePath = imagesPath + imageID
        img = cv2.imread(imagePath)
        dh, dw, _ = img.shape
        # Annotate
        prob1 = None
        action1 = None
        prob2 = None
        action2 = None
        if name == 'pedestrian':
            # pickle2 objects look like this:
            # {'frame': 146, 'pedID': '4_15_1672', 'prediction': tensor([[0.]])}
            for pred in pickle2:
                if pred['frame'] == int(frameNumber) and pred['pedID'] == item['objectID']:
                    prob1 = pred['prediction']
                    if prob1 >= 0.5:
                        confidence1 = round(((prob1-0.5)/0.5),2)
                        action1 = 'crossing'
                    else:
                        confidence1 = round(((0.5-prob1)/0.5),2)
                        action1 = 'not-crossing'
            
            for pred in pickle3:
                if pred['frame'] == int(frameNumber) and pred['pedID'] == item['objectID']:
                    prob2 = pred['prediction']
                    if prob2 >= 0.5:
                        confidence2 = round(((prob2-0.5)/0.5),2)
                        action2 = 'walking'
                    else:
                        confidence2 = round(((0.5-prob2)/0.5),2)
                        action2 = 'standing'
            
            cv2.rectangle(img, (xtl, ytl), (xbr, ybr), (36, 255, 12), 2)
            cv2.putText(img, name, (xtl, ytl-70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            # CROSS / NOT-CROSS LABELS
            if prob1 != None and action1 != None:
                actionLabel = action1 + ' ' + str(confidence1)
                cv2.putText(img, actionLabel, (xtl, ytl-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            else:
                actionLabel = 'calculating...'
                cv2.putText(img, actionLabel, (xtl, ytl-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            # WALK / STAND LABELS
            if prob2 != None and action2 != None:
                actionLabel = action2 + ' ' + str(confidence2)
                cv2.putText(img, actionLabel, (xtl, ytl-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            else:
                actionLabel = 'calculating...'
                cv2.putText(img, actionLabel, (xtl, ytl-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        elif name == 'traffic light' or name == 'crosswalk':
            cv2.rectangle(img, (xtl, ytl), (xbr, ybr), (228, 0, 224), 2)
            cv2.putText(img, name, (xtl, ytl-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (228, 0, 224), 2)
        else:
            cv2.rectangle(img, (xtl, ytl), (xbr, ybr), (169, 169, 169), 2)
            cv2.putText(img, name, (xtl, ytl-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (169,169,169), 2)
        # Replace old image with new one
        cv2.imwrite(imagePath, img)
    except Exception as e:
        pass

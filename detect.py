import numpy as np
import cv2
from config import MIN_CONF,NMS_THRESH

#MIN_CONF = 0.5
#NMS_THRESH = 0.3


def detect_people(frame, net, ln, personIdx) :

    results = []
    (H, W) = frame.shape[:2]

    blob=cv2.dnn.blobFromImage(image=frame,scalefactor=1/255,size=(416,416),swapRB=True,crop=None)

    net.setInput(blob)
    output_layer=net.forward(ln)

    boxes = []
    centroids = []
    confidences = []

    for output in output_layer:

        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]


            if classID==personIdx and MIN_CONF < confidence:

                #box = detection[0:4] * np.array([W, H, W, H])

                #center_x = int(detection[0] * W)
                #center_y = int(detection[1]* H)
                #w  = int(detection[3]*W)
                #h = int(detection[4]*H)
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")


                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)




    if len(idxs) >0:
        for i in idxs.flatten():
            (x,y,w,h) = (boxes[i][0] , boxes[i][1] ,boxes [i][2] ,boxes[i][3])
            values = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(values)


    return results










import cv2
import imutils
from detect import detect_people
from config import configPath,MIN_CONF,MIN_DISTANCE,NMS_THRESH,LABELS,weightsPath
import numpy  as np
from scipy.spatial import distance as dist
import datetime


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#vs = cv2.VideoCapture("demo video/pedestrians.mp4")
#vs = cv2.VideoCapture(0)

vs = cv2.VideoCapture("demo video/test.mp4")

ln=net.getUnconnectedOutLayersNames()


while True:


    ret, frame = vs.read()

    if not ret:
        print("Video Not Captured / Video Processing finished")
        break


    frame = imutils.resize(frame, width=1000)

    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
    #print(results)

    violations = set()

    if len(results) >=2 :
        centroids = np.array([r[2] for r in results])
        #print(centroids)
        dis = dist.cdist(centroids, centroids, metric="euclidean")

        #print(dis)
        for i in range(0, dis.shape[0]):
            for j in range(i + 1, dis.shape[1]):

                if dis[i, j] < MIN_DISTANCE:
                    violations.add(i)
                    violations.add(j)


    for (i,(probability,bbox,centroid)) in enumerate(results):
        start_X,start_Y,end_X,end_Y=bbox
        center_x,center_y = centroid
        color =(0,255,0)
        
        if i in violations:
            color=(0,0,255)
            
        cv2.rectangle(frame,(start_X,start_Y),(end_X,end_Y),color,2)
        cv2.circle(frame,(center_x,center_y),4,color,1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    datetim = str(datetime.datetime.now())

    frame = cv2.putText(frame, datetim, (0, 35), font, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)
    text = "Social Distancing Violations: {}".format(len(violations))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()


        









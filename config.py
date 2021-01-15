MIN_CONF = 0.5
NMS_THRESH = 0.3

MIN_DISTANCE = 85

labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

configPath = "yolo-coco/yolov3.cfg"
weightsPath = "yolo-coco/yolov3.weights"
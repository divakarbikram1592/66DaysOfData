import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3_training_final.weights', 'yolov3_testing.cfg')

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture('test_video.mp4')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Total frames: {frame_count}")
print(f"FPS: {fps}\n")

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size = (100, 3))
scale = 2

frames = []

for frame_index in range(frame_count):
    print(f"Frame: {frame_index + 1}")
    _, img = cap.read()
    scaled_dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img = cv2.resize(img, scaled_dim, interpolation = cv2.INTER_AREA)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB = True, crop = False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 1.25, (255, 255, 255), 2)

    frames.append(img)

    window_name = 'Mask Detection using YOLOv3'
    cv2.imshow(window_name, img)
    key = cv2.waitKey(1)

    if key == 27:
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('mask detection.mp4', fourcc, fps, (width, height))

for i in frames:
    video.write(i)

cv2.destroyAllWindows()
video.release()
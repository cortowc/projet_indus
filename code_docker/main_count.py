import cv2

from ultralytics import YOLO
import supervision as sv
from supervision import process_video
import numpy as np
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
import os
os.environ['OPENCV_VIDEOIO_BACKEND'] = 'cv2.VideoCapture'


LINE_START = sv.Point(320, 0)
LINE_END = sv.Point(320, 480)


def main():
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    video = cv2.VideoWriter('yolov8_1.avi', VideoWriter_fourcc(*'MP42'), 25.0, (640, 480))
    # Load a model
    model = YOLO("yolov8n.pt") #modèle entraîné
    #model = YOLO("yolov8n.pt") #modèle pré-entraîné
    for result in model.track(source=2, stream=True, save=True, agnostic_nms=True, tracker = "bytetrack.yaml"):
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        

        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for xyxy, _, confidence, class_id, tracker_id 
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)

        cv2.imshow("yolov8", frame)
        video.write(frame)
        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()

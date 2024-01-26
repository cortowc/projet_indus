import cv2

from ultralytics import YOLO
import supervision as sv
from supervision import process_video
import numpy as np
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
from line_counter_modified import LineZoneAnnotator, LineZone
from supervision.draw.color import Color


# Ligne  1
LINE_START_1 = sv.Point(320, 0)
LINE_END_1 = sv.Point(320, 480)

# Ligne 2 
LINE_START_2 = sv.Point(320, 0)
LINE_END_2 = sv.Point(320, 480)

# Ligne 3 
LINE_START_3 = sv.Point(320, 0)
LINE_END_3 = sv.Point(320, 480)

# Ligne 4 
LINE_START_4 = sv.Point(320, 0)
LINE_END_4 = sv.Point(320, 480)

# Ligne 5 
LINE_START_5 = sv.Point(320, 0)
LINE_END_5 = sv.Point(320, 480)


# Text annotations
# Ligne 1
in_text_x_1 = 15
in_text_y_1 = 25

out_text_x_1 = 15
out_text_y_1 = 50

# Ligne 2
in_text_x_2 = 500
in_text_y_2 = 25

out_text_x_2 = 500
out_text_y_2 = 50

# Ligne 3
in_text_x_3 = 15
in_text_y_3 = 440

out_text_x_3 = 15
out_text_y_3 = 465

# Ligne 4
in_text_x_4 = 500
in_text_y_4 = 440

out_text_x_4 = 500
out_text_y_4 = 465

# Ligne 5
in_text_x_5 = 275
in_text_y_5 = 25

out_text_x_5 = 275
out_text_y_5 = 50


# Ligne 1
in_text_position_1 = (in_text_x_1, in_text_y_1)
out_text_position_1 = (out_text_x_1, out_text_y_1)

# Ligne 2
in_text_position_2 = (in_text_x_2, in_text_y_2)
out_text_position_2 = (out_text_x_2, out_text_y_2)

# Ligne 3
in_text_position_3 = (in_text_x_3, in_text_y_3)
out_text_position_3 = (out_text_x_3, out_text_y_3)

# Ligne 4
in_text_position_4 = (in_text_x_4, in_text_y_4)
out_text_position_4 = (out_text_x_4, out_text_y_4)

# Ligne 5
in_text_position_5 = (in_text_x_5, in_text_y_5)
out_text_position_5 = (out_text_x_5, out_text_y_5)


def main():
    # LineZone : Counts the number of objects that cross a line


    line_counter_1 = LineZone(start=LINE_START_1, end=LINE_END_1)
    line_counter_2 = LineZone(start=LINE_START_2, end=LINE_END_2)
    line_counter_3 = LineZone(start=LINE_START_3, end=LINE_END_3)
    line_counter_4 = LineZone(start=LINE_START_4, end=LINE_END_4)
    line_counter_5 = LineZone(start=LINE_START_5, end=LINE_END_5)

    
    # LineZoneAnnotator : Initialize the LineCounterAnnotator object with default values (dessine les lignes de comptage et affiche les compteurs)
    line_annotator_1 = LineZoneAnnotator(thickness=2, color=Color.from_hex("#b2d897"), text_thickness=1, text_scale=0.5, custom_in_text="in cubes", custom_out_text="out cubes")

    line_annotator_2 = LineZoneAnnotator(thickness=2, color=Color.from_hex("#87ceeb"), text_thickness=1, text_scale=0.5, custom_in_text="in cylindres", custom_out_text="out cylindres")

    line_annotator_3 = LineZoneAnnotator(thickness=2, color=Color.from_hex("#f5d300"), text_thickness=1, text_scale=0.5, custom_in_text="in donuts", custom_out_text="out donuts")

    line_annotator_4 = LineZoneAnnotator(thickness=2, color=Color.from_hex("#d8b9ff"), text_thickness=1, text_scale=0.5, custom_in_text="in bonbons", custom_out_text="out bonbons")

    line_annotator_5 = LineZoneAnnotator(thickness=2, color=Color.from_hex("#DDA8A2"), text_thickness=1, text_scale=0.5, custom_in_text="in Total", custom_out_text="out Total")


    # BoxAnnotator : A class for drawing bounding boxes on an image using detections provided
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    video = cv2.VideoWriter('yolov8_4lignes_1.avi', VideoWriter_fourcc(*'MP42'), 25.0, (640, 480))
    # Load a model
    model = YOLO("/home/actemium/ultralytics/runs/detect/train11/weights/best.pt") #modèle entraîné
    #model = YOLO("yolov8n.pt")

    for result in model.track(source=2, stream=True, show=True, save = True, half=True, agnostic_nms=True, tracker = "bytetrack.yaml"):
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

	# To prevent the scrpit from crashing when we have no detections 
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        

	# Add labels to bounding boxes
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

	# Annotate : Draws bounding boxes on the frame using the detections provided
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )


        detections_1 = detections[detections.class_id==0]
        detections_2 = detections[detections.class_id==1]
        detections_3 = detections[detections.class_id==2]
        detections_4 = detections[detections.class_id==3]


	# trigger: updates the in_count and out_count for the detections that cross the line

        line_counter_1.trigger(detections=detections_1)

        line_counter_2.trigger(detections=detections_2)

        line_counter_3.trigger(detections=detections_3)

        line_counter_4.trigger(detections=detections_4)

        line_counter_5.trigger(detections=detections)


        line_annotator_1.annotate_modified(frame=frame, line_counter=line_counter_1, in_text_position =in_text_position_1, out_text_position= out_text_position_1)
        line_annotator_2.annotate_modified(frame=frame, line_counter=line_counter_2, in_text_position =in_text_position_2, out_text_position= out_text_position_2)
        line_annotator_3.annotate_modified(frame=frame, line_counter=line_counter_3, in_text_position =in_text_position_3, out_text_position= out_text_position_3)
        line_annotator_4.annotate_modified(frame=frame, line_counter=line_counter_4, in_text_position =in_text_position_4, out_text_position= out_text_position_4)
        line_annotator_5.annotate_modified(frame=frame, line_counter=line_counter_5, in_text_position =in_text_position_5, out_text_position= out_text_position_5)


        cv2.imshow("yolov8", frame)
        video.write(frame)
        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()

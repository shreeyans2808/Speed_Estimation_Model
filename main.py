import cv2
import numpy as np
from ultralytics import YOLO
import torch
import math
from collections import defaultdict

def get_M(src_points, dest_points):
    M = cv2.getPerspectiveTransform(src_points, dest_points)
    return M

def get_perspective_transform(points, M):
    points = np.array(points)
    reshaped = points.reshape(-1,1,2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(reshaped, M)
    return transformed_points.reshape(-1,2)[0][0], transformed_points.reshape(-1,2)[0][1]

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

vehicles = [0,2,3,5,7]
model = YOLO('yolo11l.pt')
video_path = '/Users/shreeyansarora/Downloads/Internship_Work/Task5/traffic_video.mp4'

# Birds Eye Information
src_points = np.array([[513, 538], [657, 535], [1068, 1201], [158, 1246]], dtype=np.float32) #Getting the values using `point_on_road.py`
dest_points = np.array([[0, 0], [25, 0], [25, 100], [0, 100]], dtype=np.float32) #points/ matrix where the source points are to be mapped into

M = get_M(src_points, dest_points)
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation_traffic.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


previous_boxes = {}
car_tracks = defaultdict(list)  # Store position history for each car
car_speeds = {}
next_car_id = 0
frame_no = 0
peak_traffic = 0
vehicles_detected = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_no += 1
    current_boxes = {}
    
    results = model(frame)
    
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box.tolist()
            if int(class_id) in vehicles:
                vehicles_detected +=1
                cx = int((x1 + x2) / 2)
                cy = int(y2) 
                bev_x, bev_y = get_perspective_transform((cx, cy), M)
                

                matched = False
                for car_id, prev_box in previous_boxes.items():
                    if calculate_iou([x1,y1,x2,y2], prev_box) > 0.5:  
                        current_boxes[car_id] = [x1,y1,x2,y2]
                        car_tracks[car_id].append((bev_x, bev_y))
                        matched = True
                        break
                
                if not matched:
                    car_id = next_car_id
                    next_car_id += 1
                    current_boxes[car_id] = [x1,y1,x2,y2]
                    car_tracks[car_id].append((bev_x, bev_y))
                
                if len(car_tracks[car_id]) >= fps:
                    positions = car_tracks[car_id][-fps:]
                    total_distance = 0
                    for i in range(len(positions)-1):
                        dx = positions[i+1][0] - positions[i][0]
                        dy = positions[i+1][1] - positions[i][1]
                        total_distance += math.sqrt(dx*dx + dy*dy)
                    

                    distance_meters = total_distance
                    speed_mps = distance_meters
                    speed_kph = speed_mps * 3.6
                    
                    car_speeds[car_id] = speed_kph
                
                speed = car_speeds.get(car_id, 0)
                color = (0, 0, 255) if speed > 60 else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"Speed: {speed:.1f} km/h", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    peak_traffic = max(peak_traffic,vehicles_detected)            
    cv2.putText(frame, f"Peak Traffic - {peak_traffic}",(700, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0 , 0), 2)
    vehicles_detected = 0
    previous_boxes = current_boxes.copy()
    
    if frame_no % 50 == 0:
        for car_id in list(car_tracks.keys()):
            if car_id not in current_boxes:
                del car_tracks[car_id]
                if car_id in car_speeds:
                    del car_speeds[car_id]
    
    #cv2.imshow('Car Speed Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    video_writer.write(frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
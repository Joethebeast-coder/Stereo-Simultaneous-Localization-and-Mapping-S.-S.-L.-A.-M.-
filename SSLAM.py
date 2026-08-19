import cv2
import numpy as np
import Calc_coords as cc
from ultralytics import YOLO


#AI Object Detection
model = YOLO("yolo26n.pt")

def collect_n_scan(left_img, right_img):
    full_pack = [] #Full Packet to be returned

    rectified_left, rectified_right = cc.rectify(left_img, right_img)
    results = model(rectified_left) #Temporary, use trained model if doesnt work

    for result in results:
        for box in result.boxes:
            packet = []

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])

            if confidence >= 0.65:
                packet.append(class_name)
                packet.append(center_x)
                packet.append(center_y)

                full_pack.append(packet)


    return rectified_left, rectified_right, full_pack

def compute_obj_dist(rectified_left, rectified_right, full_pack, Q):
    disparity_map, points_3d, valid = cc.depth_disparity(rectified_left, rectified_right, Q)
    obj_pos = []

    for packet in full_pack:
        d3_packet = []

        obj_class = packet[0]
        center_x = packet[1]
        center_y = packet[2]

        if valid[center_y, center_x]:
            obj_x, obj_y, obj_z = points_3d[center_y, center_x]
            d3_packet.append(obj_class)
            d3_packet.append(obj_x)
            d3_packet.append(obj_y)
            d3_packet.append(obj_z)

            obj_pos.append(d3_packet)

    return obj_pos, points_3d

def apriltag_pos(rectified_left, valid, points_3d):

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )
    )

    corners, ids, rejected = detector.detectMarkers(rectified_left)

    if ids is not None:
        tags = []

        for i, tag_id in enumerate(ids.flatten()):
            print("Tag ID:", tag_id)
            print("Corners:", corners[i])

            pts = corners[i][0]

            center_x = pts[:, 0].mean()
            center_y = pts[:, 1].mean()

            print(center_x, center_y)

            x = int(center_x)
            y = int(center_y)

            if valid[y, x]:
                X, Y, Z = points_3d[y, x]

            horizontal_dist = np.sqrt(X**2 + Z**2)

            tags.append(tag_id, horizontal_dist)

    else:
        return (None)



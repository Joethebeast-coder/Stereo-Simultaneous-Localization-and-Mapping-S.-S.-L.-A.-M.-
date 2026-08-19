import cv2
import numpy as np
import Calc_coords as cc
from ultralytics import YOLO

global field_width_m
global field_length_m

field_width_m = (316.64 * 2.54) / 100
field_length_m = (650.12 * 2.54) / 100

#AI Object Detection
model = YOLO("yolo26n.pt")

def collect_n_scan(left_img, right_img, map_left_x, map_left_y, map_right_x, map_right_y):
    full_pack = [] #Full Packet to be returned

    rectified_left, rectified_right = cc.rectify(left_img, right_img, map_left_x, map_left_y, map_right_x, map_right_y)
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
        center_x = int(packet[1])
        center_y = int(packet[2])

        if valid[center_y, center_x]:
            obj_x, obj_y, obj_z = points_3d[center_y, center_x]
            d3_packet.append(obj_class)
            d3_packet.append(obj_x)
            d3_packet.append(obj_y)
            d3_packet.append(obj_z)

            obj_pos.append(d3_packet)

    return obj_pos, points_3d, valid

def robot_field_pos(horizontal_dist, tag_id, r_angle_tag):

    #Ex: Tag on the top-middle (Change math as per challenge's specific apriltag positions)

    if tag_id == 1: #Example
        id_x = field_width_m / 2
        id_y = 0 #Top left - 0, 0, bottom right - (316.64 * 2.54) / 100, (650.12 * 2.54) / 100

        rr_angle_tag = np.radians(r_angle_tag)
        rob_y_dist = np.sin(rr_angle_tag * -1 if r_angle_tag < 0 else rr_angle_tag) * horizontal_dist
        rob_x_dist = np.cos(rr_angle_tag * -1 if r_angle_tag < 0 else rr_angle_tag) * horizontal_dist

        rob_x = id_x - rob_x_dist if r_angle_tag < 0 else id_x + rob_x_dist
        rob_y = rob_y_dist

        return [rob_x, rob_y]

    
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

            top_left_x     = int(pts[0][0])
            top_left_y     = int(pts[0][1])

            top_right_x    = int(pts[1][0])
            top_right_y    = int(pts[1][1])

            bottom_right_x = int(pts[2][0])
            bottom_right_y = int(pts[2][1])

            bottom_left_x  = int(pts[3][0])
            bottom_left_y  = int(pts[3][1])
            

            if valid[top_left_y, top_left_x] and valid[top_right_y, top_right_x] and valid[bottom_left_y, bottom_left_x] and valid[bottom_right_y, bottom_right_x]:
                tlx, tly, tlz = points_3d[top_left_y, top_left_x]
                trx, tRy, trz = points_3d[top_right_y, top_right_x]
                blx, bly, blz = points_3d[bottom_left_y, bottom_left_x]
                brx, bry, brz = points_3d[bottom_right_y, bottom_right_x]

                hz = np.array([trx - tlx, tRy - tly, trz - tlz])
                vz = np.array([blx - tlx, bly - tly, blz - tlz])

                normal = np.cross(hz, vz)
                normal /= np.linalg.norm(normal)

                if normal[2] < 0:
                    normal = -normal

                nx = normal[0]
                ny = normal[1]
                nz = normal[2]

                robot_angle = np.degrees(np.arctan2(nx, nz))
                

            center_x = pts[:, 0].mean()
            center_y = pts[:, 1].mean()

            print(center_x, center_y)

            x = int(center_x)
            y = int(center_y)

            if valid[y, x]:
                X, Y, Z = points_3d[y, x]

                horizontal_dist = np.sqrt(X**2 + Z**2)

                tags.append([tag_id, horizontal_dist, robot_angle])

        return tags

    else:
        return [None]


def dist_to_landmarks(rob_pos):
    rob_x = rob_pos[0]
    rob_y = rob_pos[1]

    id_1_x = field_width_m / 2
    id_1_y = 0

    id_2_x = 0
    id_2_y = field_length_m / 2

    dist_to_id_1_x = rob_x - id_1_x
    dist_to_id_1_y = rob_y - id_1_y

    dist_to_id_1 = np.sqrt(dist_to_id_1_y  ** 2 + (dist_to_id_1_x ** 2))

    dist_to_id_2_x = rob_x - id_2_x
    dist_to_id_2_y = rob_y - id_2_y

    dist_to_id_2 = np.sqrt(dist_to_id_2_y ** 2 + (dist_to_id_2_x ** 2))

    angle_id_1 = np.degrees(np.arcsin(dist_to_id_1_x / dist_to_id_1))
    angle_id_2 = np.degrees(np.arcsin(dist_to_id_2_y / dist_to_id_2))

    return [dist_to_id_1, dist_to_id_2, angle_id_1, angle_id_2]


def obstacle_settings(): #Change and adjust based on new game
    bot_radius = 0.705
    goal_length = (47 * 2.54) / 100
    goalr_center = [(182.11 * 2.54) / 100, field_width_m / 2]
    goalb_center = [((182.11 * 2.54) + (143.5 * 2 * 2.54))/ 100, field_width_m / 2]
    np.savez(
        "obs_settings.npz",
        bot_radius=bot_radius,
        goal_length=goal_length,
        goalr_center=goalr_center,
        goalb_center=goalb_center
    )


def retrieve_obs_settings(): #Change and adjust based on new game
    data = np.load("obs_settings.npz")

    return [data["bot_radius"], data["goal_length"], data["goalr_center"], data["goalb_center"]]


def determine_best_path(robot_pos, target_coords, all_obstacles):

    robot_x = robot_pos[0]
    robot_y = robot_pos[1]

    target_x = target_coords[0]
    target_y = target_coords[1]

    


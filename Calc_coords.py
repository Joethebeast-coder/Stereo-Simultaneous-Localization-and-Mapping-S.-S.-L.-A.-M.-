import cv2
import numpy as np

#Number of INNER corners in the checkerboard
CHECKERBOARD = (9, 6)

#Size of one square, in meters
SQUARE_SIZE = 0.025

#Create an array for 9*6 = 54 corners
objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

#Fill in X and Y coordinates
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

#Convert from "squares" to meters
objp *= SQUARE_SIZE

left_image_points = []
right_image_points = []
object_points = []

def calibrate(left_image, right_image):
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    left_found, left_corners = cv2.findChessboardCorners(
        left_gray,
        (9, 6)
    )

    right_found, right_corners = cv2.findChessboardCorners(
        right_gray,
        (9, 6)
    )

    if left_found and right_found:
        left_image_points.append(left_corners)
        right_image_points.append(right_corners)
        object_points.append(objp)

    image_size = left_gray.shape[::-1]

    _, K_left, D_left, _, _ = cv2.calibrateCamera(
        object_points,
        left_image_points,
        image_size,
        None,
        None
    )

    _, K_right, D_right, _, _ = cv2.calibrateCamera(
        object_points,
        right_image_points,
        image_size,
        None,
        None
    )

    flags = cv2.CALIB_FIX_INTRINSIC

    _, K_left, D_left, K_right, D_right, R, T, E, F = \
        cv2.stereoCalibrate(
            object_points,
            left_image_points,
            right_image_points,
            K_left,
            D_left,
            K_right,
            D_right,
            image_size,
            flags=flags
        )

    
    R_left, R_right, P_left, P_right, Q, roi_left, roi_right = \
        cv2.stereoRectify(
            K_left,
            D_left,
            K_right,
            D_right,
            image_size,
            R,
            T
        )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left,
        D_left,
        R_left,
        P_left,
        image_size,
        cv2.CV_32FC1
    )

    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right,
        D_right,
        R_right,
        P_right,
        image_size,
        cv2.CV_32FC1
    )

    
    return map_left_x, map_left_y, map_right_x, map_right_y, Q 


def rectify(left_image, right_image, map_left_x, map_left_y, map_right_x, map_right_y):
    rectified_left = cv2.remap(
        left_image,
        map_left_x,
        map_left_y,
        cv2.INTER_LINEAR
    )

    rectified_right = cv2.remap(
        right_image,
        map_right_x,
        map_right_y,
        cv2.INTER_LINEAR
    )

    return rectified_left, rectified_right

def save_calibration(filename, map_left_x, map_left_y, map_right_x, map_right_y, Q):
    np.savez(
        filename,
        map_left_x=map_left_x,
        map_left_y=map_left_y,
        map_right_x=map_right_x,
        map_right_y=map_right_y,
        Q=Q
    )

def load_calibration(filename):
    data = np.load(filename)

    return (
        data["map_left_x"],
        data["map_left_y"],
        data["map_right_x"],
        data["map_right_y"],
        data["Q"]
    )

def depth_disparity(rectified_left, rectified_right, Q):
    left_gray = cv2.cvtColor(
        rectified_left,
        cv2.COLOR_BGR2GRAY
    )

    right_gray = cv2.cvtColor(
        rectified_right,
        cv2.COLOR_BGR2GRAY
    )

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5
    )

    disparity_map = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    valid = disparity_map > 0
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)


    return disparity_map, points_3d, valid



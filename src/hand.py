import cv2
import numpy as np
import time

# Open Camera object
cap = cv2.VideoCapture(0)

# Decrease frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600);


def draw(img, corners, imgpts,action):
    imgpts = np.int32(imgpts).reshape(-1,2)
    imgpts = imgpts+(action*10)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

def calibrate(img):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # Find the rotation and translation vectors.
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return corners2,ret, mtx, dist, rvecs, tvecs


def augment(img, corners2, rvecs, tvecs, mtx, dist, action):
    axis = np.float32([[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0],
                       [0, 0, -2], [0, 2, -2], [2, 2, -2], [2, 0, -2]])
    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, np.asarray(rvecs), np.asarray(tvecs), mtx, dist)
    img = draw(img, corners2, imgpts,action)
    cv2.imshow('img', img)


def project_augmentation(action):
    # action = 0
    CHESS_BOARD = 'camera_calibration.jpg'
    img = cv2.imread(CHESS_BOARD)
    corners2, ret, mtx, dist, rvecs, tvecs = calibrate(img)
    augment(img, corners2, np.asarray(rvecs), np.asarray(tvecs), mtx, dist, action)



def nothing(x):
    pass


# Function to find angle between two vectors
def Angle(v1, v2):
    dot = np.dot(v1, v2)
    x_modulus = np.sqrt((v1 * v1).sum())
    y_modulus = np.sqrt((v2 * v2).sum())
    cos_angle = dot / x_modulus / y_modulus
    angle = np.degrees(np.arccos(cos_angle))
    return angle


# Function to find distance between two points in a list of lists
def FindDistance(A, B):
    return np.sqrt(np.power((A[0][0] - B[0][0]), 2) + np.power((A[0][1] - B[0][1]), 2))


# Creating a window for HSV track bars
cv2.namedWindow('HSV_TrackBar')

# Starting with 100's to prevent error while masking
h, s, v = 100, 100, 100

# Creating track bar
cv2.createTrackbar('h', 'HSV_TrackBar', 0, 179, nothing)
cv2.createTrackbar('s', 'HSV_TrackBar', 0, 255, nothing)
cv2.createTrackbar('v', 'HSV_TrackBar', 0, 255, nothing)

while (1):
    try:
        # Measure execution time
        start_time = time.time()

        # Capture frames from the camera
        ret, frame = cap.read()

        # Blur the image
        blur = cv2.blur(frame, (3, 3))

        # Convert to HSV color space
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))

        # Kernel matrices for morphological transformation
        kernel_square = np.ones((11, 11), np.uint8)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Perform morphological transformations to filter out the background noise
        # Dilation increase skin color area
        # Erosion increase skin color area
        dilation = cv2.dilate(mask2, kernel_ellipse, iterations=1)
        erosion = cv2.erode(dilation, kernel_square, iterations=1)
        dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
        filtered = cv2.medianBlur(dilation2, 5)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
        median = cv2.medianBlur(dilation2, 5)
        ret, thresh = cv2.threshold(median, 127, 255, 0)

        # Find contours of the filtered frame
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw Contours
        # cv2.drawContours(frame, cnt, -1, (122,122,0), 3)
        # cv2.imshow('Dilation',median)

        # Find Max contour area (Assume that hand is in the frame)
        max_area = 100
        ci = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if (area > max_area):
                max_area = area
                ci = i

                # Largest area contour
        cnts = contours[ci]

        # Find convex hull
        hull = cv2.convexHull(contours[ci])

        # Find convex defects
        hull2 = cv2.convexHull(contours[ci], returnPoints=False)
        defects = cv2.convexityDefects(contours[ci], hull2)

        # Get defect points and draw them in the original image
        FarDefect = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contours[ci][s][0])
            end = tuple(contours[ci][e][0])
            far = tuple(contours[ci][f][0])
            FarDefect.append(far)
            cv2.line(frame, start, end, [0, 255, 0], 1)
            cv2.circle(frame, far, 10, [100, 255, 255], 3)

        # Find moments of the largest contour
        moments = cv2.moments(contours[ci])

        # Central mass of first order moments
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
        centerMass = (cx, cy)

        # Draw center mass
        cv2.circle(frame, centerMass, 7, [100, 0, 255], 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Center', tuple(centerMass), font, 2, (255, 255, 255), 2)

        # Distance from each finger defect(finger webbing) to the center mass
        distanceBetweenDefectsToCenter = []
        for i in range(0, len(FarDefect)):
            x = np.array(FarDefect[i])
            centerMass = np.array(centerMass)
            distance = np.sqrt(np.power(x[0] - centerMass[0], 2) + np.power(x[1] - centerMass[1], 2))
            distanceBetweenDefectsToCenter.append(distance)

        # Get an average of three shortest distances from finger webbing to center mass
        sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
        AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])

        # Get fingertip points from contour hull
        # If points are in proximity of 80 pixels, consider as a single point in the group
        finger = []
        for i in range(0, len(hull) - 1):
            if (np.absolute(hull[i][0][0] - hull[i + 1][0][0]) > 80) or (
                np.absolute(hull[i][0][1] - hull[i + 1][0][1]) > 80):
                if hull[i][0][1] < 500:
                    finger.append(hull[i][0])

        # The fingertip points are 5 hull points with largest y coordinates
        finger = sorted(finger, key=lambda x: x[1])
        fingers = finger[0:5]

        # Calculate distance of each finger tip to the center mass
        fingerDistance = []
        for i in range(0, len(fingers)):
            distance = np.sqrt(np.power(fingers[i][0] - centerMass[0], 2) + np.power(fingers[i][1] - centerMass[0], 2))
            fingerDistance.append(distance)

        # Finger is pointed/raised if the distance of between fingertip to the center mass is larger
        # than the distance of average finger webbing to center mass by 130 pixels
        result = 0
        for i in range(0, len(fingers)):
            if fingerDistance[i] > AverageDefectDistance + 130:
                result = result + 1

        # Print number of pointed fingers
        cv2.putText(frame, str(result), (100, 100), font, 2, (255, 255, 255), 2)
        project_augmentation(result)
        # show height raised fingers
        # cv2.putText(frame,'finger1',tuple(finger[0]),font,2,(255,255,255),2)
        # cv2.putText(frame,'finger2',tuple(finger[1]),font,2,(255,255,255),2)
        # cv2.putText(frame,'finger3',tuple(finger[2]),font,2,(255,255,255),2)
        # cv2.putText(frame,'finger4',tuple(finger[3]),font,2,(255,255,255),2)
        # cv2.putText(frame,'finger5',tuple(finger[4]),font,2,(255,255,255),2)
        # cv2.putText(frame,'finger6',tuple(finger[5]),font,2,(255,255,255),2)
        # cv2.putText(frame,'finger7',tuple(finger[6]),font,2,(255,255,255),2)
        # cv2.putText(frame,'finger8',tuple(finger[7]),font,2,(255,255,255),2)

        # Print bounding rectangle
        x, y, w, h = cv2.boundingRect(contours[ci])
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.drawContours(frame, [hull], -1, (255, 255, 255), 2)

        ##### Show final image ########
        cv2.imshow('Dilation', frame)
        ###############################

        # Print execution time
        # print time.time()-start_time

        # close the output video by pressing 'ESC'
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    except:
        continue

cap.release()
cv2.destroyAllWindows()
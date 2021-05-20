import cv2
import time
import PoseModule as pm
import mediapipe as mp
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--fps", type=int, default=30,
	help="set fps of output video")
args = vars(ap.parse_args())


pTime = 0
black_flag = False
cap = cv2.VideoCapture(args["input"])
out = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(*"MJPG"), 
                      args["fps"], (int(cap.get(3)), int(cap.get(4))))

detector = pm.PoseDetector()

while(cap.isOpened()):
    success, img = cap.read()
    
    if success == False:
        break
    
    img, p_landmarks, p_connections = detector.findPose(img, False)
    
    # use black background
    if black_flag:
        img = img * 0
    
    # draw points
    mp.solutions.drawing_utils.draw_landmarks(img, p_landmarks, p_connections)
    lmList = detector.getPosition(img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    out.write(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()

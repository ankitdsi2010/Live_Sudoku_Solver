from main import getCellPositions, extractSudokuDigits, detectEmptyCell, placeSudokuDigitsLive
from algo import solve
from utils import four_point_transform, find_area, preprocessImage
import cv2
import numpy as np

# Connects to computer's default camera
capture = cv2.VideoCapture(0)

# Automatically grab width and height from video
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# to run lines once
flag = True

while True:
    # capture frame by frame
    ret, frame = capture.read()
    # convert into gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find area contour
    poly_approx = find_area(gray)
    # transform image
    board_segment = four_point_transform(gray, poly_approx)
    # Clearing the image
    thresh_inv = preprocessImage(board_segment)
    # Applying Probabilistic Hough Transform
    lines = cv2.HoughLinesP(thresh_inv, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    for l in lines:
        x1, y1, x2, y2 = l[0]
        cv2.line(board_segment, (x1,y1), (x2,y2), (0,255,0), 2, cv2.LINE_AA)
    if flag:
        a = extractSudokuDigits(thresh_inv)
        solve(a)
        placeSudokuDigitsLive(thresh_inv)
        flag = False
    # overlaying board segment of image on frame
    x_offset, y_offset = (poly_approx[0][0].tolist()[0]), (poly_approx[0][0].tolist()[1])
    x_end, y_end = (x_offset+board_segment.shape[1]), (y_offset+board_segment.shape[0])
    frame[y_offset:y_end, x_offset:x_end] = board_segment
    # display the frame
    if ret:
        cv2.startWindowThread()
        cv2.namedWindow("Window")
        cv2.imshow("Window", frame)
    # quit by pressing "q" on keyboard
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# When everthing is done, destroy all windows
capture.release()
cv2.destroyAllWindows()

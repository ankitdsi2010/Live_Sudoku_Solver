from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
from algo import solve
import cv2

new_model = load_model('keras_digit_model.h5')

# model to predict digit
def prediction(test_image):
    classes = new_model.predict_classes(test_image)
    if classes == [[0]]:
        return 0
    elif classes == [[1]]:
        return 1
    elif classes == [[2]]:
        return 2
    elif classes == [[3]]:
        return 3
    elif classes == [[4]]:
        return 4
    elif classes == [[5]]:
        return 5
    elif classes == [[6]]:
        return 6
    elif classes == [[7]]:
        return 7
    elif classes == [[8]]:
        return 8
    elif classes == [[9]]:
        return 9

# creating a function to get the different position of cells
def getCellPositions(img_PT):
    # resizing image
    img_PT = cv2.resize(img_PT, (252,252))
    # computing position of each cell
    cell_positions = []
    width = img_PT.shape[1]
    height = img_PT.shape[0]
    cell_width = width / 9
    cell_height = height / 9
    x1, x2, y1, y2 = 0, 0, 0, 0
    for _ in range(9):
        y2 = y1 + cell_height
        x1 = 0
        for _ in range(9):
            x2 = x1 + cell_width
            current_cell = [x1, x2, y1, y2]
            cell_positions.append(current_cell)
            x1 = x2
        y1 = y2
    return cell_positions

# using the trained model to predict the digits in each cell
def predictDigit(cell, img):
    pos = []
    img = cv2.resize(img, (252,252))
    img = img[int(cell[2]+2):int(cell[3]-3),int(cell[0]+2):int(cell[1]-3)]
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            # if contour is large, it is a digit
            if (w < 15 and x > 2) and (h < 25 and y > 2):
                pos.append((x,y,x+w,y+h))
                break
    if pos == []:
        result = 0
    if pos:
        img1 = img[(pos[0][1]):(pos[0][3]), (pos[0][0]):(pos[0][2])]
        img1 = cv2.resize(img1, (28,28))
        img1 = img1.reshape(1,28,28,1)
        img1 = tf.cast(img1, tf.float32)
        result = prediction(img1)
    return result

def extractSudokuDigits(img_PT):
    # we look at the middle of cell where digit should be placed
    cell_digits, num = [], 0
    cells = getCellPositions(img_PT)
    for cell in range(len(cells)):
        num = predictDigit(cells[cell],img_PT)
        cell_digits.append(num)
    n = 9
    cell_digits = [cell_digits[i:i+n] for i in range(0, len(cell_digits), n)]
    return cell_digits

def detectEmptyCell(cell, img):
    pos = []
    img = cv2.resize(img, (252,252))
    img = img[int(cell[2]+2):int(cell[3]-3),int(cell[0]+2):int(cell[1]-3)]
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # if container is large, it must be a digit
            if (w < 15 and x > 2) and (h < 25 and y > 2):
                pos.append((x,y,x+w,y+h))
                break
    if pos == []:
        return pos
    else:
        return 0

def placeSudokuDigitsLive(img_g, img):
    # We look at the middle of the cell
    img_PT = cv2.resize(img_g, (252,252))
    img_color = cv2.resize(img, (252,252))
    cells = getCellPositions(img_PT)
    n = 9
    cell_rs = [cells[i:i+n] for i in range(0,len(cells),n)]
    digits = extractSudokuDigits(img_PT)
    solve(digits)
    for i in range(len(cell_rs)):
        for j in range(len(cell_rs[i])):
            pos = detectEmptyCell(cell_rs[i][j], img_PT)
            digit_text = digits[i][j]
            if pos == []:
                cv2.putText(img_color, str(digit_text), ((cell_rs[i][j][0]+8),(cell_rs[i][j][2]+19)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
            else:
                continue




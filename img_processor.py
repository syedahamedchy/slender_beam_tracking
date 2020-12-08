import cv2
import imutils
import numpy as np

def img_processor(im0, r):
    gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)  # Convert to gray scale.
    blur0 = cv2.blur(gray0, (5, 5))  # Image blurring.
    ret, bw0 = cv2.threshold(blur0, 52, 255, cv2.THRESH_BINARY) #Binarise image.

    def is_contour_bad(c):
        if ((cv2.contourArea(c) > 250) or (cv2.contourArea(c) < 15)):
            return True
        else:
            return False

    # Now we loop over all contours to form a mask layer, which is then used to form the final image.
    cnts = cv2.findContours(bw0.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    mask = np.ones(im0.shape[:2], dtype="uint8") * 255

    for c in cnts:
        if is_contour_bad(c):
            cv2.drawContours(mask, [c], -1, 0, -1)
    bw0 = cv2.bitwise_and(bw0, bw0, mask=mask)

    img_cropped = bw0[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    cnts = cv2.findContours(img_cropped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    X = []
    Y = []

    for c in cnts:
        M = cv2.moments(c)
        if (M["m00"] == 0):
            pass
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            X.append(cX)
            Y.append(cY)
    return X, Y

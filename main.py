import cv2
import glob 
import time
import imutils
import numpy as np
import pandas as pd
from numpy import savetxt
import matplotlib.pyplot as plt
from img_processor import img_processor
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance
import math

# --- Process and display the reference image to set up a region of interest (ROI). --- #

##First we process the initial image (including making the image greyscale, blurring it and finally binarising it)
##to make it ready for processing. Let's first read the image into a variable. 
img_dir = 'Cam2_sinus_7.2Hz_400mV/*.bmp'
imname = glob.glob(img_dir)[0]
im0 = cv2.imread(imname)

gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)  # Convert to gray scale.
blur0 = cv2.blur(gray0, (5, 5))  # Image blurring.
th = 52  #First set a threshold, then apply a function to binarise the image with that threshold.  
ret, bw0 = cv2.threshold(blur0, th, 255, cv2.THRESH_BINARY)

##Now we have to filter out some of the irrelevant binarised pixels from the image 
##(e.g the larger white region due to the cantilever support)

#Define a function to say that sufficiently large contours are unwanted in our final image. 
#Additionally, very small contours are removed to avoid noise from entering the image. 
def is_contour_bad(c):
    if ((cv2.contourArea(c) > 250) or (cv2.contourArea(c) < 15)):
        return True
    else:
        return False

#Now we loop over all contours to form a mask layer, which is then used to form the final image. 
cnts = cv2.findContours(bw0.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
mask = np.ones(im0.shape[:2], dtype="uint8") * 255

for c in cnts:
	if is_contour_bad(c):
		cv2.drawContours(mask, [c], -1, 0, -1)
bw0 = cv2.bitwise_and(bw0, bw0, mask=mask)

##Now form the region of interest (ROI) which all images will be cropped into.
cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL) #For resizing purposes
cv2.waitKey(0)
r = cv2.selectROI("Cropped", bw0) #The variable 'r' will be in the form of the following list: [x, y, width, height]
cv2.waitKey(0)
imCrop = bw0[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
cv2.imshow("Cropped", imCrop)  #Display cropped image.
cv2.waitKey(0)






# --- Set up analyse every frame and extract centroid coordinates. --- #

X_tracker = []
Y_tracker = []
Tracker_Data = []

for i in range (0, len(glob.glob(img_dir)), 1):
    imname = glob.glob(img_dir)[i]
    im = cv2.imread(imname)
    X, Y = img_processor(im, r)
    X_tracker.append(X)
    Y_tracker.append(Y)
    #Tracker_Data.append(X)
    #Tracker_Data.append(Y)

pd.DataFrame(X_tracker).to_csv("x_raw.csv")
pd.DataFrame(Y_tracker).to_csv("y_raw.csv")
#print(Tracker_Data)

#print(X_tracker)
#print(Y_tracker)
#print(data[0])
#plt.scatter(X_tracker[0], Y_tracker[0])
#plt.title('Slender Beam Movement')
#plt.show()






# --- Display animation to visualise data. --- #

fig, axs = plt.subplots()
axs.set_xlim((0, 3000))
axs.set_ylim((0, 1500))
   
beam, = axs.plot( X_tracker[0], Y_tracker[0], 'xb-')

def animate(i):
    targetx = np.array(X_tracker[i])
    targety = np.array(Y_tracker[i])
    beam.set_data( [targetx, targety])
    return beam,
raw_animation = FuncAnimation(fig, animate, frames=len(X_tracker), blit=True)
plt.show()

raw_animation.save('7.2Hz_400mV1.gif', writer='pillow', fps=15)

X_scaled = []
Y_scaled = []

for i in range (0, len(glob.glob(img_dir)), 1):
    f = np.polyfit(X_tracker[i], Y_tracker[i], deg=5)
    x = np.linspace(min(X_tracker[i]), max(X_tracker[i]), 1000) #This is not robust for if the beam folds in on itself.
    y = np.polyval(f, x) 
    
    X_scaled.append(x)
    Y_scaled.append(y)

fig, axs = plt.subplots()
axs.set_xlim((0, 3000))
axs.set_ylim((0, 1500))
   
beam, = axs.plot( X_scaled[0], Y_scaled[0])

def animate1(i):
    targetx = np.array(X_scaled[i])
    targety = np.array(Y_scaled[i])
    beam.set_data( [targetx, targety])
    return beam,
scaled_animation = FuncAnimation(fig, animate1, frames=len(X_scaled), blit=True)
plt.show()

#scaled_animation.save('7.2Hz_400mV2.gif', writer='pillow', fps=15)

pd.DataFrame(X_scaled).to_csv("x_regfit.csv")
pd.DataFrame(Y_scaled).to_csv("y_regfit.csv")

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
from scipy import signal 
from scipy.interpolate import interp1d
import math
import os

# --- Process and display the reference image to set up a region of interest (ROI). --- #
##First we process the initial image (including making the image greyscale, blurring it and finally binarising it)
##to make it ready for processing. Let's first read the image into a variable. 
test_frequency = 6.8
test_voltage = 600
folder_name = 'Cam2_sinus_' + str(test_frequency) + 'Hz_' + str(test_voltage) + 'mV'
img_dir = 'D:/Data/' + folder_name + '/*.bmp'

if not os.path.exists('D:/Data/Results/' + folder_name):
    os.makedirs('D:/Data/Results/' + folder_name)
results_path = 'D:/Data/Results/' + folder_name

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

#pd.DataFrame(X_tracker).to_csv("x_raw.csv")
#pd.DataFrame(Y_tracker).to_csv("y_raw.csv")







# --- Display animation to visualise data. --- #

fig, axs = plt.subplots()
axs.set_xlim((0, 3000))
axs.set_ylim((0, 1500))
   
beam, = axs.plot( X_tracker[0], Y_tracker[0], 'o')

def animate(i):
    targetx = np.array(X_tracker[i])
    targety = np.array(Y_tracker[i])
    beam.set_data( [targetx, targety])
    return beam,
raw_animation = FuncAnimation(fig, animate, frames=len(X_tracker), blit=True)
#plt.show()

raw_animation.save(results_path + '/raw.gif', writer='pillow', fps=15)

X_scaled = []
Y_scaled = []

for i in range (0, len(glob.glob(img_dir)), 1):
    f = np.polyfit(X_tracker[i], Y_tracker[i], deg=5)
    x = np.linspace(min(X_tracker[i]), max(X_tracker[i]), 1000)
    y = np.polyval(f, x) 
    
    X_scaled.append(x)
    Y_scaled.append(y)

pd.DataFrame(X_scaled).to_csv(results_path + '/x_raw.csv')
pd.DataFrame(Y_scaled).to_csv(results_path + '/y_raw.csv')




# --- Split up regression fitted data into equal segments. --- #
# Get the fitted data.
x_raw = pd.read_csv(results_path + '/x_raw.csv')
x_raw = x_raw.drop(x_raw.columns[[0]], axis=1)
x0 = np.mean(x_raw.iloc[:, 0])
x_raw = x_raw - x0

y_raw = pd.read_csv(results_path + '/y_raw.csv')
y_raw = y_raw.drop(y_raw.columns[[0]], axis=1) 
y0 = np.mean(y_raw.iloc[:, 0])
y_raw = y_raw - y0

# Calculate length of reference frame and the length of each segment.
x0 = x_raw.iloc[0,:]
y0 = y_raw.iloc[0,:]

L = 0
for i in range(0, (len(x0)-1), 1):
    L = L + math.dist((x0[i], y0[i]), (x0[i+1], y0[i+1]))

n = 20  #Number of datapoints to form.
dL = L/(n + 1)

X_scaled = []
Y_scaled = []

for i in range (1, len(x_raw), 1):
    x = x_raw.iloc[i,:]
    y = y_raw.iloc[i,:]

    X_fit = []
    Y_fit = []

    L0 = 0
    k = 1

    for j in range(0, (len(x)-1), 1):
        L0 = L0 + math.dist((x[j], y[j]), (x[j+1], y[j+1]))
        if (L0>=(k*dL)):
            X_fit.append(x[j])
            Y_fit.append(y[j])
            k = k + 1

    X_scaled.append(X_fit)
    Y_scaled.append(Y_fit)

# fig, axs = plt.subplots()
# axs.set_xlim((0, 3000))
# axs.set_ylim((-500, 500))
# beam, = axs.plot(X_scaled[0], Y_scaled[0], '-o')

# def animate1(i):
#     targetx = np.array(X_scaled[i])
#     targety = np.array(Y_scaled[i])
#     beam.set_data([targetx, targety])
#     return beam,

# scaled_animation = FuncAnimation(fig, animate1, frames=len(X_scaled), blit=True)
# scaled_animation.save(results_path + '/regression.gif', writer='pillow', fps=15)

pd.DataFrame(X_scaled).drop(labels=n, axis=1).to_csv(results_path + '/xregression_fit.csv')
pd.DataFrame(Y_scaled).drop(labels=n, axis=1).to_csv(results_path + '/yregression_fit.csv')







# -- Smooth out the data with a moving average filter ##

# Get the fitted data.
x_raw = pd.read_csv(results_path + '/xregression_fit.csv')
x_raw = np.array(x_raw.drop(x_raw.columns[[0]], axis=1))

y_raw = pd.read_csv(results_path + '/yregression_fit.csv')
y_raw = np.array(y_raw.drop(y_raw.columns[[0]], axis=1))

t_raw = pd.read_csv('D:/Data/' + folder_name + '/Sequencetime.csv', delimiter=';')
t_raw = np.array(t_raw.iloc[:,0])
t_raw = np.delete(t_raw, 0)
t_raw = np.delete(t_raw, 0)
t_raw = t_raw.astype(np.float)

t=[float(0)]
time=0

# Form a gif to show the regression data, before any filtering is applied.
for i in range(0, (len(t_raw) - 1), 1):
    time = time + (t_raw[i+1] -  t_raw[i])
    t.append(time)

fig, axs = plt.subplots()
axs.set_xlim((0, 3000))
axs.set_ylim((-500, 500))
beam, = axs.plot(x_raw[0], y_raw[0], '-o')

def animate1(i):
    targetx = np.array(x_raw[i])
    targety = np.array(y_raw[i])
    beam.set_data([targetx, targety])
    return beam,

scaled_animation = FuncAnimation(fig, animate1, frames=len(x_raw), blit=True)
scaled_animation.save(results_path + '/regression.gif', writer='pillow', fps=15)



# Plot the time series data of the tip: unfiltered as well as low, medium and heavy filters. 
figure, axis = plt.subplots(4, 2, figsize=(15, 15))
plt.subplots_adjust(hspace=1)

axis[0, 0].set_title('X Tip Coordinate Unfiltered')
axis[0, 0].plot(t, x_raw[:, -1])
axis[0, 0].set_xlabel('Time (s)')
axis[0, 0].set_ylabel('Position (Pixels)')

axis[0, 1].set_title('Y Tip Coordinate Unfiltered')
axis[0, 1].plot(t, y_raw[:, -1])
axis[0, 1].set_xlabel('Time (s)')
axis[0, 1].set_ylabel('Position (Pixels)')

# Now apply moving averages to smooth out the noise in the data, 
# assuming that the sampling frequency is much higher than the 
# relevant frequences of the beam.

x_time_averaged = np.zeros(shape=((len(x_raw) - 2), len(x_raw[0])))
y_time_averaged = np.zeros(shape=((len(y_raw) - 2), len(y_raw[0])))
for i in range(1, (len(x_raw) - 1), 1):
    for j in range(0, len(x_raw[0, :]), 1):
        x_time_averaged[(i - 1), j] = (x_raw[(i - 1), j] + x_raw[i, j] + x_raw[(i + 1), j])/3
        y_time_averaged[(i - 1), j] = (y_raw[(i - 1), j] + y_raw[i, j] + y_raw[(i + 1), j])/3

x0 = x_time_averaged[:, -1]
y0 = y_time_averaged[:, -1]

axis[1, 0].set_title('X Tip Coordinate Low Filter (k = 1)')
axis[1, 0].plot(t[1:-1], x0)
axis[1, 0].set_xlabel('Time (s)')
axis[1, 0].set_ylabel('Position (Pixels)')

axis[1, 1].set_title('Y Tip Coordinate Low Filter (k = 1)')
axis[1, 1].plot(t[1:-1], y0)
axis[1, 1].set_xlabel('Time (s)')
axis[1, 1].set_ylabel('Position (Pixels')

#pd.DataFrame(x_time_averaged).to_csv(results_path + '/x_smoothed.csv')
#pd.DataFrame(y_time_averaged).to_csv(results_path + '/y_smoothed.csv')
#pd.DataFrame(t[1:-1]).to_csv(results_path + '/time.csv')


x_time_averaged = np.zeros(shape=((len(x_raw) - 4), len(x_raw[0])))
y_time_averaged = np.zeros(shape=((len(y_raw) - 4), len(y_raw[0])))
for i in range(2, (len(x_raw) - 2), 1):
    for j in range(0, len(x_raw[0, :]), 1):
        x_time_averaged[(i - 2), j] = (x_raw[(i - 2), j] + x_raw[(i - 1), j] + x_raw[(i), j] + x_raw[(i + 1), j] + x_raw[(i + 2), j])/5
        y_time_averaged[(i - 2), j] = (y_raw[(i - 2), j] + y_raw[(i - 1), j] + y_raw[(i), j] + y_raw[(i + 1), j] + y_raw[(i + 2), j])/5

x0 = x_time_averaged[:, -1]
y0 = y_time_averaged[:, -1]

axis[2, 0].set_title('X Tip Coordinate Medium Filter (k = 2)')
axis[2, 0].plot(t[2:-2], x0)
axis[2, 0].set_xlabel('Time (s)')
axis[2, 0].set_ylabel('Position (Pixels)')

axis[2, 1].set_title('Y Tip Coordinate Medium Filter (k = 2)')
axis[2, 1].plot(t[2:-2], y0)
axis[2, 1].set_xlabel('Time (s)')
axis[2, 1].set_ylabel('Position (Pixels)')

pd.DataFrame(x_time_averaged).to_csv(results_path + '/x_smoothed.csv')
pd.DataFrame(y_time_averaged).to_csv(results_path + '/y_smoothed.csv')
pd.DataFrame(t[1:-1]).to_csv(results_path + '/time.csv')


x_time_averaged = np.zeros(shape=((len(x_raw) - 12), len(x_raw[0])))
y_time_averaged = np.zeros(shape=((len(y_raw) - 12), len(y_raw[0])))
for i in range(6, (len(x_raw) - 6), 1):
    for j in range(0, len(x_raw[0, :]), 1):
        x_time_averaged[(i - 6), j] = (x_raw[(i - 6), j] + x_raw[(i - 5), j] + x_raw[(i - 4), j] + x_raw[(i - 3), j] + x_raw[(i - 2), j] + x_raw[(i - 1), j] + x_raw[(i), j] + x_raw[(i + 1), j] + x_raw[(i + 2), j] + x_raw[(i + 3), j] + x_raw[(i + 4), j] + x_raw[(i + 5), j] + x_raw[(i + 6), j])/13
        y_time_averaged[(i - 6), j] = (y_raw[(i - 6), j] + y_raw[(i - 5), j] + y_raw[(i - 4), j] + y_raw[(i - 3), j] + y_raw[(i - 2), j] + y_raw[(i - 1), j] + y_raw[(i), j] + y_raw[(i + 1), j] + y_raw[(i + 2), j] + y_raw[(i + 3), j] + y_raw[(i + 4), j] + y_raw[(i + 5), j] + y_raw[(i + 6), j])/13

x0 = x_time_averaged[:, -1]
y0 = y_time_averaged[:, -1]

axis[3, 0].set_title('X Tip Coordinate Heavy (k = 3)')
axis[3, 0].plot(t[6:-6], x0)
axis[3, 0].set_xlabel('Time (s)')
axis[3, 0].set_ylabel('Position (Pixels)')

axis[3, 1].set_title('Y Tip Coordinate Heavy Filter (k = 3)')
axis[3, 1].plot(t[6:-6], y0)
axis[3, 1].set_xlabel('Time (s)')
axis[3, 1].set_ylabel('Position (Pixels)')

#pd.DataFrame(x_time_averaged).to_csv(results_path + '/x_smoothed.csv')
#pd.DataFrame(y_time_averaged).to_csv(results_path + '/y_smoothed.csv')
#pd.DataFrame(t[1:-1]).to_csv(results_path + '/time.csv')

plt.savefig(results_path + '/filters.png')

# Animation of the filtered data.
fig, axs = plt.subplots()
axs.set_xlim((0, 3000))
axs.set_ylim((-500, 500))
   
beam, = axs.plot(x_time_averaged[0], y_time_averaged[0], 'o-')

def animate2(i):
    targetx = np.array(x_time_averaged[i])
    targety = np.array(y_time_averaged[i])
    beam.set_data( [targetx, targety])
    return beam,
smoothed_animation = FuncAnimation(fig, animate2, frames=len(x_time_averaged), blit=True)

smoothed_animation.save(results_path + '/smoothed.gif', writer='pillow', fps=15)





## Apply Fourier Analysis
# First get time data.
t_raw = pd.read_csv('D:/Data/' + folder_name + '/Sequencetime.csv', delimiter=';')
t_raw = np.array(t_raw.iloc[:,0])
t_raw = np.delete(t_raw, 0)
t_raw = np.delete(t_raw, 0)
t_raw = t_raw.astype(np.float)

t=[float(0)]
time=0

for i in range(0, (len(t_raw) - 1), 1):
    time = time + (t_raw[i+1] - t_raw[i])
    t.append(time)

#Adjust the time series to ensure that its the same length as the smoothed dataset. 
#t = t[1:-1]    # Low Filter
t = t[2:-2]    # Medium Filter
#t = t[6:-6]    # Heavy Filter
t_period = np.mean(np.diff(t))

# Form interpolated time domain.
t_interpolated = np.linspace(t[0], t[-1], 2500)


# Get the fitted spatial data.
x_raw = pd.read_csv(results_path + '/x_smoothed.csv') 
#x_raw = pd.read_csv(results_path + '/xregression_fit.csv') #(If you want an unfiltered analysis)
x_raw = np.array(x_raw.drop(x_raw.columns[[0]], axis=1))

y_raw = pd.read_csv(results_path + '/y_smoothed.csv') 
#y_raw = pd.read_csv(results_path + '/yregression_fit.csv') #(If you want an unfiltered analysis)
y_raw = np.array(y_raw.drop(y_raw.columns[[0]], axis=1))

x0 = x_raw[:, 0]
x0 = x0 - np.mean(x0)
#f = interp1d(t, x0, kind='cubic')
#x0 = f(t_interpolated)

x1 = x_raw[:, (math.floor(len(x_raw[0])/2) - 1)]
x1 = x1 - np.mean(x1)
#f = interp1d(t, x1, kind='cubic')
#x1 = f(t_interpolated)

x2 = x_raw[:, -1]
x2 = x2 - np.mean(x2)
#f = interp1d(t, x2, kind='cubic')
#x2 = f(t_interpolated)

y0 = y_raw[:, 0]
y0 = y0 - np.mean(y0)
#f = interp1d(t, y0, kind='cubic')
#y0 = f(t_interpolated)

y1 = y_raw[:, (math.floor(len(y_raw[0])/2) - 1)]
y1 = y1 - np.mean(y1)
#f = interp1d(t, y1, kind='cubic')
#y1 = f(t_interpolated)

y2 = y_raw[:, -1]
y2 = y2 - np.mean(y2)
#f = interp1d(t, y2, kind='cubic')
#y2 = f(t_interpolated)


# Create subplot
figure, axis = plt.subplots(4, 2, figsize=(15,15))
plt.subplots_adjust(hspace=1)
x_range = 30

axis[0, 0].set_title('X Coordinate of the Tip')
axis[0, 0].plot(t, x2)
axis[0, 0].set_xlabel('Time')
axis[0, 0].set_ylabel('Amplitude')

axis[0, 1].set_title('Y Coordinate of the Tip')
axis[0, 1].plot(t, y2)
axis[0, 1].set_xlabel('Time')
axis[0, 1].set_ylabel('Amplitude')

#Discrete Frequency Transformations
amplitude = np.pad(x2, (0,0), 'constant')
fourierTransform = np.fft.fft(amplitude)/len(amplitude)           # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(amplitude)/2))] # Exclude sampling frequency
#fourierTransform[0:1] = np.mean(fourierTransform[-10:])           # Ignore aliasing caused by sampling frequency
tpCount     = len(amplitude)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/(1/t_period)
frequencies = values/timePeriod

axis[1, 0].set_title('Discrete Fourier Transform of X Coordinate at Tip')
axis[1, 0].plot(frequencies, abs(fourierTransform))
axis[1, 0].set_xlabel('Frequency (Hz)')
axis[1, 0].set_ylabel('Amplitude of X')

amplitude = np.pad(y2, (0,0), 'constant')
fourierTransform = np.fft.fft(amplitude)/len(amplitude)           # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(amplitude)/2))] # Exclude sampling frequency
#fourierTransform[0:1] = np.mean(fourierTransform[-10:])           # Ignore aliasing caused by sampling frequency
tpCount     = len(amplitude)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/(1/t_period)
frequencies = values/timePeriod

axis[1, 1].set_title('Discrete Fourier Transform of Y Coordinate at Tip')
axis[1, 1].plot(frequencies, abs(fourierTransform))
axis[1, 1].set_xlabel('Frequency (Hz)')
axis[1, 1].set_ylabel('Amplitude of Y')

peak_freq = frequencies[np.argmax(fourierTransform)]
axis[1, 1].text(s='Peak Frequency is at:' + str(round(peak_freq, 2)) + 'Hz', x=25, y = 25)

amplitude = np.pad(x1, (0,0), 'constant')
fourierTransform = np.fft.fft(amplitude)/len(amplitude)           # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(amplitude)/2))] # Exclude sampling frequency
#fourierTransform[0:1] = np.mean(fourierTransform[-10:])           # Ignore aliasing caused by sampling frequency
tpCount     = len(amplitude)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/(1/t_period)
frequencies = values/timePeriod

axis[2, 0].set_title('Discrete Fourier Transform of X Coordinate at Midpoint')
axis[2, 0].plot(frequencies, abs(fourierTransform))
axis[2, 0].set_xlabel('Frequency (Hz)')
axis[2, 0].set_ylabel('Amplitude of Y')

amplitude = np.pad(y1, (0,0), 'constant')
fourierTransform = np.fft.fft(amplitude)/len(amplitude)           # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(amplitude)/2))] # Exclude sampling frequency
#fourierTransform[0:1] = np.mean(fourierTransform[-10:])           # Ignore aliasing caused by sampling frequency
tpCount     = len(amplitude)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/(1/t_period)
frequencies = values/timePeriod

axis[2, 1].set_title('Discrete Fourier Transform of Y Coordinate at Midpoint')
axis[2, 1].plot(frequencies, abs(fourierTransform))
axis[2, 1].set_xlabel('Frequency (Hz)')
axis[2, 1].set_ylabel('Amplitude of Y')

peak_freq = frequencies[np.argmax(fourierTransform)]
axis[2, 1].text(s='Peak Frequency is at:' + str(round(peak_freq, 2)) + 'Hz', x=25, y = 25)

amplitude = np.pad(x0, (0,0), 'constant')
fourierTransform = np.fft.fft(amplitude)/len(amplitude)           # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(amplitude)/2))] # Exclude sampling frequency
#fourierTransform[0:1] = np.mean(fourierTransform[-10:])           # Ignore aliasing caused by sampling frequency
tpCount     = len(amplitude)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/(1/t_period)
frequencies = values/timePeriod

axis[3, 0].set_title('Discrete Fourier Transform of X Coordinate at Root')
axis[3, 0].plot(frequencies, abs(fourierTransform))
axis[3, 0].set_xlabel('Frequency (Hz)')
axis[3, 0].set_ylabel('Amplitude of Y')

amplitude = np.pad(y0, (0,0), 'constant')
fourierTransform = np.fft.fft(amplitude)/len(amplitude)           # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(amplitude)/2))] # Exclude sampling frequency
#fourierTransform[0:1] = np.mean(fourierTransform[-10:])           # Ignore aliasing caused by sampling frequency
tpCount     = len(amplitude)
values      = np.arange(int(tpCount/2))
timePeriod  = tpCount/(1/t_period)
frequencies = values/timePeriod

axis[3, 1].set_title('Discrete Fourier Transform of Y Coordinate at Root')
axis[3, 1].plot(frequencies, abs(fourierTransform))
axis[3, 1].set_xlabel('Frequency (Hz)')
axis[3, 1].set_ylabel('Amplitude of Y')

peak_freq = frequencies[np.argmax(fourierTransform)]
axis[3, 1].text(s='Peak Frequency is at:' + str(round(peak_freq, 2)) + 'Hz', x=5, y = 5)

plt.savefig(results_path + '/DFT.png')
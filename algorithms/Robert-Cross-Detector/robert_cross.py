import tkinter as tk
import tkinter.filedialog as fd
import cv2
import numpy as np
from scipy.signal import convolve2d

def convolve(image, kernel):
    convolved_matrix = convolve2d(image, kernel, mode='same', boundary='symm')
    return np.array(convolved_matrix, dtype=np.float64)

def robert_cross_algorithm(img):
    # Apply Robert Cross Gx kernel
    x_gradient_matrix = convolve(img, gx_kernel)
    # Apply Robert Cross Gy kernel
    y_gradient_matrix = convolve(img, gy_kernel)
    # Combine the gradient matrices to get the gradient magnitude
    gradient_matrix = np.sqrt(x_gradient_matrix**2 + y_gradient_matrix**2)
    return gradient_matrix / np.max(gradient_matrix) * 255

def on_tr_trackbar(val):
    global gradient_image, threshold_tracker, img
    gradient_magnitude = robert_cross_algorithm(img)

    threshold_tracker = int(cv2.getTrackbarPos('Threshold', 'Trackbars'))
    
    gradient_image = np.zeros(gradient_magnitude.shape, dtype=np.uint8)
    gradient_image[gradient_magnitude > threshold_tracker] = 255
    gradient_image = cv2.putText(gradient_image, f'T: {threshold_tracker}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show results
    cv2.imshow('Gray Scale', img)
    cv2.imshow('Robert Cross Operator', gradient_image)

IMAGE_FILE = '../../test-images/simple-shapes.png'
IMAGE_NAME = IMAGE_FILE.split('/')[-1]
img = cv2.imread(IMAGE_FILE, 0)

threshold_tracker = 0

gradient_image = None

gx_kernel = np.array([[-1, 0],
                      [0, 1]])

gy_kernel = np.array([[0, -1],
                      [1, 0]])

#Creating trackbar window
cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 640, 240)
cv2.createTrackbar("Threshold", "Trackbars", 0, 255, on_tr_trackbar)

on_tr_trackbar(50)

cv2.waitKey(0)
cv2.destroyAllWindows()

gradient_image = cv2.putText(gradient_image, f'T: {threshold_tracker}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.imwrite('results/' + IMAGE_NAME, gradient_image)


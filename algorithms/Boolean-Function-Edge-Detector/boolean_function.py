from statistics import variance
import tkinter as tk
import tkinter.filedialog as fd
import cv2
import numpy as np
from scipy.signal import convolve2d

boolean_functions_matrices = [
    np.array([[0, 1, 1],
              [0, 1, 1],
              [0, 1, 1]]),
    np.array([[0, 0, 0],
              [1, 1, 1],
              [1, 1, 1]]),
    np.array([[1, 1, 0],
              [1, 1, 0],
              [1, 1, 0]]),
    np.array([[1, 1, 1],
              [1, 1, 1],
              [0, 0, 0]]),
    
    np.array([[1, 1, 1],
              [0, 1, 1],
              [0, 0, 1]]),
    np.array([[1, 0, 0],
              [1, 1, 0],
              [1, 1, 1]]),
    np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 1, 1]]),
    np.array([[1, 1, 1],
              [1, 1, 0],
              [1, 0, 0]]),
    
    np.array([[0, 1, 1],
              [0, 1, 1],
              [0, 0, 1]]),
    np.array([[1, 0, 0],
              [1, 1, 0],
              [1, 1, 0]]),
    np.array([[0, 0, 0],
              [0, 1, 1],
              [1, 1, 1]]),
    np.array([[1, 1, 1],
              [1, 1, 0],
              [0, 0, 0]]),

    np.array([[1, 1, 1],
              [0, 1, 1],
              [0, 0, 0]]),
    np.array([[0, 0, 0],
              [1, 1, 0],
              [1, 1, 1]]),
    np.array([[0, 0, 1],
              [0, 1, 1],
              [0, 1, 1]]),
    np.array([[1, 1, 0],
              [1, 1, 0],
              [1, 0, 0]]),
]

def perform_boolean_functions(image, window_size=3, c=0):
    new_img = np.zeros(image.shape, dtype=np.uint8)
    for i in range(1,image.shape[0]):
        for j in range(1,image.shape[1]):
            # Calculate the local window
            local_window = image[i-window_size//2:i+window_size//2 + 1, j-window_size//2:j+window_size//2 + 1].copy()
            # Calculate the mean and standard deviation
            threshold = np.mean(local_window) - c
            # Compare local window value with threshold
            local_window[local_window > threshold] = 255
            local_window[local_window <= threshold] = 0
            # Check if local window is equal to a boolean function
            local_window //= 255
            for boolean_function in boolean_functions_matrices:
                if np.all(local_window == boolean_function):
                    new_img[i, j] = 255
                    break
    return new_img

def calculate_variance(image, window_size=3):
    new_img = np.zeros(image.shape, dtype=np.uint8)
    for i in range(1,image.shape[0]):
        for j in range(1,image.shape[1]):
            # Calculate the local window
            local_window = image[i-window_size//2:i+window_size//2 + 1, j-window_size//2:j+window_size//2 + 1].copy()
            variance_local_window = np.var(local_window)
            # Calculate the mean and standard deviation
            new_img[i, j] = variance_local_window
    return new_img

def threshold(img, threshold):
    new_img = np.zeros(img.shape, dtype=np.uint8)
    new_img[img > threshold] = 255
    return new_img

def boolean_function_algorithm(img, window_size=3, c=0, global_threshold=0):
    global boolean_function_image, variance_image
    boolean_function_image = perform_boolean_functions(img, window_size=window_size, c = c)
    variance_image = calculate_variance(img, window_size=3)
    global_threshold = int(global_threshold * np.max(variance_image) / 500)
    global_thresholded_image = threshold(variance_image, global_threshold)
    final_image = boolean_function_image & global_thresholded_image
    final_image = cv2.putText(final_image, f'C: {c} T: {global_threshold}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Show results
    cv2.imshow('Boolean Function', boolean_function_image)
    cv2.imshow('Global Threshold', global_thresholded_image)
    cv2.imshow('Final Image', final_image)

def on_c_trackbar(val):
    c = cv2.getTrackbarPos('C', 'Trackbars')
    thresh = cv2.getTrackbarPos('T', 'Trackbars')
    boolean_function_algorithm(img, window_size=3, c=c, global_threshold=thresh)

def on_thresh_trackbar(val):
    global img, variance_image
    c = cv2.getTrackbarPos('C', 'Trackbars')
    variance_image = calculate_variance(img, window_size=3)
    thresh = int(cv2.getTrackbarPos('T', 'Trackbars') * np.max(variance_image) / 500)
    global_thresholded_image = threshold(variance_image, thresh)
    final_image = boolean_function_image & global_thresholded_image
    final_image = cv2.putText(final_image, f'C: {c} T: {thresh}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('Global Threshold', global_thresholded_image)
    cv2.imshow('Final Image', final_image)

IMAGE_PATH = '../../test-images/lena.png'
img = cv2.imread(IMAGE_PATH, 0)
boolean_function_image = None
variance_image = None
cv2.imshow('Image', img)

cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 640, 240)
cv2.createTrackbar("C", "Trackbars", 0, 255, on_c_trackbar)
cv2.createTrackbar("T", "Trackbars", 0, 500, on_thresh_trackbar)

on_c_trackbar(1)
on_thresh_trackbar(1)

cv2.waitKey(0)
cv2.destroyAllWindows()

# # Save image
# cv2.imwrite('results/' + img_name, gradient_image)
from typing import final
import cv2
import numpy as np
from scipy.signal import convolve2d

def convolve(image, kernel):
    convolved_matrix = convolve2d(image, kernel, mode='same', boundary='symm')
    return convolved_matrix

def any_neighbor_has_value(img, i, j, value=0):
    k_start = 0 if j - 1 < 0 else -1
    k_end  = 0 if j + 1 > img.shape[1] - 1 else 1
    l_start = 0 if i - 1 < 0 else -1
    l_end  = 0 if i + 1 > img.shape[0] - 1 else 1
    for k in range(k_start, k_end):
      for l in range(l_start, l_end):
         if img[i+l, j+k] == value:
            return True
    return False
    
def zero_crossing(image):
    z_c_image = np.zeros(image.shape)
    
    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i+1, j-1],image[i+1, j],image[i+1, j+1],image[i, j-1],image[i, j+1],image[i-1, j-1],image[i-1, j],image[i-1, j+1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1
 
 
            # If both negative and positive values exist in 
            # the pixel neighborhood, then that pixel is a 
            # potential zero crossing
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            # Change the pixel value with the maximum neighborhood
            # difference with the pixel
 
            if z_c:
                if image[i,j]>0:
                    z_c_image[i, j] = image[i,j] + np.abs(e)
                elif image[i,j]<0:
                    z_c_image[i, j] = np.abs(image[i,j]) + d
 
    return np.uint8(z_c_image / np.max(z_c_image) * 255)

def double_threshold_image(img, low_thresh, high_thresh, weak_value=70, strong_value=255):
    # Thresholding the image
    strong = np.where(img > high_thresh)
    zeros = np.where(img < low_thresh)
    weak = np.where((img <= high_thresh) & (img >= low_thresh))

    new_image = np.zeros(img.shape, dtype=np.uint8)
    new_image[strong] = strong_value
    new_image[weak] = weak_value
    new_image[zeros] = 0
    return new_image

def is_valid_pixel(img, i, j):
    return i < img.shape[0] and i >= 0 and j < img.shape[1] and j >= 0

def valid_neighbors(img, i, j):
    neighbors = []
    for k in range(-1, 2 , 1):
        for p in range(-1, 2, 1):
            if is_valid_pixel(img, i + k, j + p):
                neighbors.append((i + k, j + p))
    return neighbors
    
def hysteresis_tracking(img, weak_value=70, strong_value=255):
    img_row, img_col = img.shape
    
    new_img = img.copy()
    for i in range(img_row):
        for j in range(img_col):
            if new_img[i,j] == strong_value:
                queue = []
                for neighbor in valid_neighbors(new_img, i, j):
                    if new_img[neighbor[0], neighbor[1]] == weak_value:
                        queue.append(neighbor)
                while len(queue) > 0:
                    x, y = queue.pop(0)
                    if new_img[x,y] == weak_value:
                        new_img[x,y] = strong_value
                        for neighbor in valid_neighbors(new_img, x, y):
                            if new_img[neighbor[0], neighbor[1]] == weak_value:
                                queue.append(neighbor)

    for i in range(img_row):
        for j in range(img_col):
            if new_img[i,j] == weak_value:
                new_img[i,j] = 0

    return new_img

def LoG_seperated(img):
    global log_image, final_image, kernel_tracker, sigma_tracker, low_threshold_tracker, high_threshold_tracker
    #Smoothing the image using gaussian kernel
    if sigma_tracker != 0:
        blurred_image = cv2.GaussianBlur(img, (kernel_tracker, kernel_tracker), sigma_tracker)
    else:
        blurred_image = img
    cv2.imshow('Gaussian Blur', blurred_image)
    # Apply laplacian kernel
    kernel = np.array([[1, 1, 1],
                        [1, -8, 1],
                        [1, 1, 1]])
    
    log_image = zero_crossing(convolve(blurred_image, kernel))
    cv2.imshow('zero crossing', log_image)
    l_image = zero_crossing(convolve(img, kernel))
    cv2.imshow('zero crossing original', l_image)
    # Thresholding the image
    thresh_image = double_threshold_image(log_image, low_threshold_tracker, high_threshold_tracker)
    cv2.imshow('Thresholded image', thresh_image)
    final_image = hysteresis_tracking(thresh_image)
    final_image = cv2.putText(final_image, f'LT: {low_threshold_tracker} HT: {high_threshold_tracker} K: {kernel_tracker}, Sig: {sigma_tracker}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Show results
    cv2.imshow('Final image', final_image)

def handle_trackbar():
    global img, threshold_tracker, kernel_tracker, sigma_tracker
    sigma_tracker = int(cv2.getTrackbarPos('Sigma', 'Trackbars'))
    kernel_tracker = int(cv2.getTrackbarPos('K', 'Trackbars') + 1) * 2 + 1
    LoG_seperated(img)

def on_sigma_trackbar(val):
    handle_trackbar()

def on_kernel_trackbar(val):
    handle_trackbar()

def handle_threshold_trackbars():
    global log_image, threshold_tracker, final_image
    low_threshold_tracker = int(cv2.getTrackbarPos('Low Threshold', 'Trackbars'))
    high_threshold_tracker =int(cv2.getTrackbarPos('High Threshold', 'Trackbars'))

    thresh_image = double_threshold_image(log_image, low_threshold_tracker, high_threshold_tracker)
    final_image = hysteresis_tracking(thresh_image)
    final_image = cv2.putText(final_image, f'LT: {low_threshold_tracker} HT: {high_threshold_tracker} K: {kernel_tracker}, Sig: {sigma_tracker}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('Thresholded image', thresh_image)
    cv2.imshow('Final image', final_image)

def on_low_thresh_trackbar(val):
    handle_threshold_trackbars()

def on_high_thresh_trackbar(val):
    handle_threshold_trackbars()



IMAGE_FILE = '../../../test-images/lena-noisy.jpeg'
IMAGE_NAME = IMAGE_FILE.split('/')[-1]
img = cv2.imread(IMAGE_FILE, 0)
log_image = img
final_image = img
low_threshold_tracker = 0
high_threshold_tracker = 0
sigma_tracker = 0
kernel_tracker = 1

#Creating trackbar window
cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 640, 240)
cv2.createTrackbar("K", "Trackbars", 1, 20, on_kernel_trackbar)
cv2.createTrackbar("Sigma", "Trackbars", 1, 24, on_sigma_trackbar)
cv2.createTrackbar("Low Threshold", "Trackbars", 1, 255, on_low_thresh_trackbar)
cv2.createTrackbar("High Threshold", "Trackbars", 1, 255, on_high_thresh_trackbar)


on_sigma_trackbar(1)
on_kernel_trackbar(1)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save image
final_image = cv2.putText(final_image, f'LT: {low_threshold_tracker} HT: {high_threshold_tracker} K: {kernel_tracker}, Sig: {sigma_tracker}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.imwrite('results/' + IMAGE_NAME, final_image)
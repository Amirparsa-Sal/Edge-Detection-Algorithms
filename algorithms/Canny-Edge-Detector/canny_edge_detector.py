from typing import final
import numpy as np
from scipy.signal import convolve2d
import cv2

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

def is_valid_pixel(img, i, j):
    return i < img.shape[0] and i >= 0 and j < img.shape[1] and j >= 0

def valid_neighbors(img, i, j):
    neighbors = []
    for k in range(-1, 2 , 1):
        for p in range(-1, 2, 1):
            if is_valid_pixel(img, i + k, j + p):
                neighbors.append((i + k, j + p))
    return neighbors

def double_threshold_image(img, low_thresh, high_thresh, weak_value=100, strong_value=255):
    highThreshold = img.max() * high_thresh;
    lowThreshold = highThreshold * low_thresh;

    # Thresholding the image
    strong = np.where(img >= highThreshold)
    zeros = np.where(img < lowThreshold)
    weak = np.where((img < highThreshold) & (img >= lowThreshold))

    new_image = np.zeros(img.shape, dtype=np.uint8)
    new_image[strong] = strong_value
    new_image[weak] = weak_value
    new_image[zeros] = 0
    return new_image

def hysteresis_tracking(img, weak_value=100, strong_value=255):
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

def convolve(image, kernel):
    convolved_matrix = convolve2d(image, kernel, mode='same', boundary='symm')
    return np.array(convolved_matrix, dtype=np.float64)

def sobel_algorithm(img):
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    G = np.hypot(Ix, Iy)
    G = np.array(G / G.max() * 255, dtype=np.uint8)
    theta = np.arctan2(Iy, Ix)
    return (G, theta)

def non_max_supression(gradient_magnitude, edge_directions):
    M, N = gradient_magnitude.shape
    angle = edge_directions * 180. / np.pi
    angle[angle < 0] += 180

    new_img = np.zeros((M,N), dtype=np.uint8)
    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 255
            r = 255
            
            #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = gradient_magnitude[i, j+1]
                r = gradient_magnitude[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = gradient_magnitude[i+1, j-1]
                r = gradient_magnitude[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = gradient_magnitude[i+1, j]
                r = gradient_magnitude[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = gradient_magnitude[i-1, j-1]
                r = gradient_magnitude[i+1, j+1]

            if (gradient_magnitude[i,j] >= q) and (gradient_magnitude[i,j] >= r):
                new_img[i,j] = gradient_magnitude[i,j]
    return new_img


def canny_algorithm(img):
    global max_edges_image, low_thresh_tracker, high_thresh_tracker, sigma_tracker, kernel_size_tracker
    # Apply Gaussian blur
    blurred_image = None
    if sigma_tracker != 0:
        blurred_image = cv2.GaussianBlur(img, (kernel_size_tracker, kernel_size_tracker), sigma_tracker)
    else:
        blurred_image = img
    # Apply Sobel algorithm
    gradient_magnitude, gradient_direction = sobel_algorithm(blurred_image)
    # Apply non maximum suppression
    max_edges_image = non_max_supression(gradient_magnitude, gradient_direction)
    cv2.imshow("Blurred", blurred_image)
    cv2.imshow("Sobel", gradient_magnitude)
    cv2.imshow("Non max", max_edges_image)

    handle_thresholds(max_edges_image)

def handle_bluring():
    canny_algorithm(img)    

def on_kernel_trackbar(val):
    global kernel_size_tracker
    kernel_size_tracker = (cv2.getTrackbarPos('K', 'Trackbars') + 1) * 2 + 1
    handle_bluring()

def on_sigma_trackbar(val):
    global sigma_tracker
    sigma_tracker = cv2.getTrackbarPos('Sigma', 'Trackbars') / 10
    handle_bluring()

def handle_thresholds(max_edges_image):
    global sigma_tracker, kernel_size_tracker, low_thresh_tracker, high_thresh_tracker, final_image
    # Apply double thresholding
    thresholded_image = double_threshold_image(max_edges_image, low_thresh_tracker, high_thresh_tracker)
    # Apply hysteresis tracking
    final_image = hysteresis_tracking(thresholded_image)
    final_image = cv2.putText(final_image, f'LT: {low_thresh_tracker} HT: {high_thresh_tracker} K: {kernel_size_tracker}, Sig: {sigma_tracker}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Show results
    cv2.imshow('Threshold image', thresholded_image)
    cv2.imshow('Final image', final_image)


def on_low_thresh_trackbar(val):
    global max_edge_image, low_thresh_tracker
    low_thresh_tracker = cv2.getTrackbarPos('Low Threshold', 'Trackbars') / 100
    handle_thresholds(max_edges_image)

def on_high_thresh_trackbar(val):
    global max_edge_image, high_thresh_tracker
    high_thresh_tracker = cv2.getTrackbarPos('High Threshold', 'Trackbars') / 100
    handle_thresholds(max_edges_image)

IMAGE_FILE = '../../test-images/bal.jpg'
IMAGE_NAME = IMAGE_FILE.split('/')[-1]
img = cv2.imread(IMAGE_FILE, 0)
max_edges_image = None
kernel_size_tracker = 25
sigma_tracker = 4
low_thresh_tracker = 0.7
high_thresh_tracker = 0.17
final_image = None

#Creating trackbar window
cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 640, 240)
cv2.createTrackbar("K", "Trackbars", 1, 20, on_kernel_trackbar)
cv2.createTrackbar("Sigma", "Trackbars", 1, 240, on_sigma_trackbar)
cv2.createTrackbar("Low Threshold", "Trackbars", 1, 100, on_low_thresh_trackbar)
cv2.createTrackbar("High Threshold", "Trackbars", 1, 100, on_high_thresh_trackbar)

canny_algorithm(img)  

# cv2.imshow('OpenCV Canny', cv2.Canny(img, 100, 140))
cv2.waitKey(0)
cv2.destroyAllWindows()

final_image = cv2.putText(final_image, f'LT: {low_thresh_tracker} HT: {high_thresh_tracker} K: {kernel_size_tracker}, Sig: {sigma_tracker}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.imwrite('results/' + IMAGE_NAME, final_image)
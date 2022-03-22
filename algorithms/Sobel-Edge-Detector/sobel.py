import tkinter as tk
import tkinter.filedialog as fd
import cv2
import numpy as np
from scipy.signal import convolve2d

def convolve_and_scale_to_255(image, kernel):
    convolved_matrix = np.abs(convolve2d(image, kernel, mode='same', boundary='symm'))
    return np.array((convolved_matrix / np.max(convolved_matrix)) * 255, dtype=np.float64)

#Loading the image
root = tk.Tk()
root.withdraw()
img_path = fd.askopenfilename()
root.destroy()
img_name = img_path.split('/')[-1]
img = cv2.imread(img_path, 0)

# Apply Sobel Gx kernel
gx_kernel = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])
x_gradient_matrix = convolve_and_scale_to_255(img, gx_kernel)

# Apply Sobel Gy kernel
gy_kernel = np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])
y_gradient_matrix = convolve_and_scale_to_255(img, gy_kernel)

# Combine the two gradient matrices
gradient_image = np.sqrt(x_gradient_matrix**2 + y_gradient_matrix**2)
gradient_image = np.array((gradient_image / np.max(gradient_image)) * 255, dtype=np.uint8)

# Show results
cv2.imshow('Gray Scale', img)
cv2.imshow('Sobel Operator', gradient_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save image
cv2.imwrite('results/' + img_name, gradient_image)
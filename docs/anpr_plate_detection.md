# ANPR Plate Detection Engine: Step-by-Step Notebook

This notebook demonstrates the main steps for license plate detection based on [ANPR Series Part 2: Advanced Plate Detection Engine](https://henok.cloud/articles/anpr-part-2-plate-detection/).

**Outline:**
1. Import Required Libraries
2. Load and Preprocess the Input Image (using Part 1 pipeline)
3. Detect Contours
4. Geometric Filtering and Aspect Ratio Validation
5. Edge Density and Texture Analysis
6. Visualize Detected Plate Candidates
7. (Optional) Benchmarking and Performance Analysis

---

## 1. Import Required Libraries
We will use OpenCV, NumPy, and Matplotlib, and define helper classes for plate detection.


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

# For notebook display
%matplotlib inline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## 2. Load and Preprocess the Input Image

We will load an image and preprocess it using the pipeline from Part 1. Make sure the preprocessing notebook or function is available in your environment.


```python
# Load the image (replace with your image path)
image_path = './sample_images/sample_image6.jpg'  # Update this path as needed
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}. Please check the path.")

# Display the original image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.title('Original Input Image')
plt.axis('off')
plt.show()

# Import the preprocessing function from Part 1 (ensure it's available)
# from anpr_image_preprocessing import preprocess_for_anpr
# For demonstration, we'll use a simple threshold as a placeholder:
preprocessed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
preprocessed = cv2.adaptiveThreshold(preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=(8, 6))
plt.imshow(preprocessed, cmap='gray')
plt.title('Preprocessed Image (Binary)')
plt.axis('off')
plt.show()
```


    
![png](anpr_plate_detection_files/anpr_plate_detection_3_0.png)
    



    
![png](anpr_plate_detection_files/anpr_plate_detection_3_1.png)
    


## 3. Detect Contours

Find contours in the preprocessed (binary) image. These contours are potential candidates for license plates.


```python
# Find contours in the preprocessed image
contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours on the image for visualization
contour_img = image_rgb.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

plt.figure(figsize=(8, 6))
plt.imshow(contour_img)
plt.title('All Contours Detected')
plt.axis('off')
plt.show()

print(f"Number of contours found: {len(contours)}")
```


    
![png](anpr_plate_detection_files/anpr_plate_detection_5_0.png)
    


    Number of contours found: 288


## 4. Geometric Filtering and Aspect Ratio Validation

Filter contours based on geometric properties (area, aspect ratio, size) to find likely plate candidates.


```python
# Loosened geometric constraints for debugging
min_area = 200  # was 800
max_area = 60000  # was 45000
min_width = 40   # was 120
max_width = 1000 # was 600
min_height = 10  # was 25
max_height = 300 # was 150
aspect_ratio_range = (2.0, 8.0)  # was (3.0, 6.5)

candidates = []
print("Contour debug info (x, y, w, h, area, aspect_ratio):")
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    aspect_ratio = w / h if h > 0 else 0
    print(f"  ({x}, {y}, {w}, {h}, {area:.1f}, {aspect_ratio:.2f})")
    if (min_area <= area <= max_area and
        min_width <= w <= max_width and
        min_height <= h <= max_height and
        aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
        candidates.append((x, y, w, h, aspect_ratio, area, contour))

# Draw candidate bounding boxes
candidate_img = image_rgb.copy()
for (x, y, w, h, aspect_ratio, area, contour) in candidates:
    cv2.rectangle(candidate_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

plt.figure(figsize=(8, 6))
plt.imshow(candidate_img)
plt.title('Geometric Filtered Plate Candidates')
plt.axis('off')
plt.show()

print(f"Number of geometric candidates: {len(candidates)}")
```

    Contour debug info (x, y, w, h, area, aspect_ratio):
      (211, 479, 1, 1, 0.0, 1.00)
      (166, 477, 4, 3, 3.0, 1.33)
      (524, 459, 2, 2, 0.5, 1.00)
      (538, 417, 1, 1, 0.0, 1.00)
      (535, 415, 3, 2, 0.0, 1.50)
      (535, 413, 1, 1, 0.0, 1.00)
      (491, 401, 3, 3, 1.0, 1.00)
      (634, 343, 6, 1, 0.0, 6.00)
      (61, 301, 1, 1, 0.0, 1.00)
      (48, 301, 1, 1, 0.0, 1.00)
      (25, 295, 6, 1, 0.0, 6.00)
      (80, 294, 67, 3, 17.0, 22.33)
      (172, 291, 2, 2, 1.0, 1.00)
      (220, 290, 3, 1, 0.0, 3.00)
      (213, 289, 1, 1, 0.0, 1.00)
      (1, 284, 1, 1, 0.0, 1.00)
      (386, 254, 2, 1, 0.0, 2.00)
      (12, 252, 1, 1, 0.0, 1.00)
      (88, 251, 1, 1, 0.0, 1.00)
      (73, 249, 3, 1, 0.0, 3.00)
      (297, 248, 1, 1, 0.0, 1.00)
      (84, 248, 1, 2, 0.0, 0.50)
      (390, 246, 1, 2, 0.0, 0.50)
      (387, 245, 1, 1, 0.0, 1.00)
      (3, 242, 1, 1, 0.0, 1.00)
      (388, 238, 2, 1, 0.0, 2.00)
      (0, 238, 1, 1, 0.0, 1.00)
      (390, 236, 2, 1, 0.0, 2.00)
      (387, 232, 1, 1, 0.0, 1.00)
      (54, 231, 1, 1, 0.0, 1.00)
      (130, 230, 96, 9, 214.5, 10.67)
      (386, 226, 2, 2, 0.5, 1.00)
      (297, 225, 5, 6, 15.0, 0.83)
      (386, 224, 1, 1, 0.0, 1.00)
      (377, 224, 7, 11, 22.0, 0.64)
      (381, 221, 1, 1, 0.0, 1.00)
      (55, 219, 1, 1, 0.0, 1.00)
      (0, 218, 2, 3, 0.0, 0.67)
      (4, 211, 50, 40, 1512.5, 1.25)
      (0, 210, 11, 7, 31.0, 1.57)
      (326, 209, 1, 3, 0.0, 0.33)
      (386, 208, 2, 1, 0.0, 2.00)
      (344, 208, 1, 1, 0.0, 1.00)
      (384, 207, 1, 1, 0.0, 1.00)
      (0, 205, 8, 3, 7.0, 2.67)
      (360, 204, 1, 1, 0.0, 1.00)
      (354, 204, 1, 1, 0.0, 1.00)
      (350, 204, 1, 1, 0.0, 1.00)
      (362, 200, 4, 2, 0.5, 2.00)
      (343, 200, 4, 7, 8.0, 0.57)
      (349, 199, 7, 3, 3.5, 2.33)
      (80, 199, 1, 1, 0.0, 1.00)
      (107, 198, 2, 1, 0.0, 2.00)
      (76, 198, 3, 1, 0.0, 3.00)
      (272, 195, 1, 1, 0.0, 1.00)
      (65, 195, 170, 48, 5262.0, 3.54)
      (380, 194, 1, 1, 0.0, 1.00)
      (355, 194, 1, 1, 0.0, 1.00)
      (382, 191, 1, 1, 0.0, 1.00)
      (374, 190, 6, 3, 6.5, 2.00)
      (366, 183, 12, 13, 31.5, 0.92)
      (270, 182, 1, 1, 0.0, 1.00)
      (20, 182, 1, 1, 0.0, 1.00)
      (273, 180, 6, 4, 1.5, 1.50)
      (240, 180, 1, 1, 0.0, 1.00)
      (373, 179, 1, 3, 0.0, 0.33)
      (270, 179, 42, 15, 267.5, 2.80)
      (42, 177, 1, 1, 0.0, 1.00)
      (36, 177, 2, 1, 0.0, 2.00)
      (268, 176, 8, 3, 12.0, 2.67)
      (44, 176, 1, 1, 0.0, 1.00)
      (33, 176, 1, 3, 0.0, 0.33)
      (0, 174, 32, 22, 407.0, 1.45)
      (369, 172, 6, 3, 4.0, 2.00)
      (303, 172, 8, 5, 16.0, 1.60)
      (172, 171, 1, 1, 0.0, 1.00)
      (167, 171, 1, 1, 0.0, 1.00)
      (194, 169, 4, 1, 0.0, 4.00)
      (372, 168, 1, 2, 0.0, 0.50)
      (358, 168, 2, 1, 0.0, 2.00)
      (215, 168, 2, 1, 0.0, 2.00)
      (229, 164, 8, 4, 2.5, 2.00)
      (222, 164, 3, 4, 4.0, 0.75)
      (231, 163, 1, 1, 0.0, 1.00)
      (241, 161, 23, 5, 28.5, 4.60)
      (306, 159, 8, 4, 11.0, 2.00)
      (22, 154, 288, 36, 2587.0, 8.00)
      (364, 153, 9, 23, 36.5, 0.39)
      (304, 152, 14, 5, 12.5, 2.80)
      (192, 151, 2, 1, 0.0, 2.00)
      (367, 150, 1, 1, 0.0, 1.00)
      (369, 147, 1, 4, 0.0, 0.25)
      (316, 146, 20, 71, 238.0, 0.28)
      (365, 144, 2, 1, 0.0, 2.00)
      (185, 143, 1, 1, 0.0, 1.00)
      (368, 140, 3, 5, 7.0, 0.60)
      (189, 140, 1, 1, 0.0, 1.00)
      (370, 136, 1, 1, 0.0, 1.00)
      (381, 135, 1, 1, 0.0, 1.00)
      (386, 134, 2, 2, 0.5, 1.00)
      (382, 133, 2, 1, 0.0, 2.00)
      (164, 133, 1, 1, 0.0, 1.00)
      (501, 130, 4, 1, 0.0, 4.00)
      (496, 129, 2, 1, 0.0, 2.00)
      (363, 129, 1, 1, 0.0, 1.00)
      (109, 129, 1, 1, 0.0, 1.00)
      (520, 127, 1, 1, 0.0, 1.00)
      (495, 127, 1, 1, 0.0, 1.00)
      (101, 127, 2, 1, 0.0, 2.00)
      (421, 126, 3, 3, 3.5, 1.00)
      (104, 126, 1, 1, 0.0, 1.00)
      (97, 126, 3, 2, 0.5, 1.50)
      (522, 125, 9, 3, 10.0, 3.00)
      (414, 125, 10, 9, 32.0, 1.11)
      (386, 125, 27, 13, 77.5, 2.08)
      (379, 125, 14, 5, 22.0, 2.80)
      (125, 125, 38, 6, 75.5, 6.33)
      (110, 125, 16, 3, 11.0, 5.33)
      (107, 125, 2, 2, 1.0, 1.00)
      (534, 124, 22, 4, 46.0, 5.50)
      (500, 124, 8, 5, 15.0, 1.60)
      (368, 124, 1, 1, 0.0, 1.00)
      (601, 122, 2, 1, 0.0, 2.00)
      (517, 122, 4, 2, 1.0, 2.00)
      (547, 121, 1, 1, 0.0, 1.00)
      (539, 121, 2, 1, 0.0, 2.00)
      (528, 121, 9, 1, 0.0, 9.00)
      (409, 121, 4, 3, 3.5, 1.33)
      (576, 120, 7, 4, 10.5, 1.75)
      (357, 119, 1, 1, 0.0, 1.00)
      (552, 118, 21, 4, 20.0, 5.25)
      (391, 118, 14, 9, 56.5, 1.56)
      (372, 118, 17, 9, 18.5, 1.89)
      (603, 116, 1, 1, 0.0, 1.00)
      (483, 115, 1, 2, 0.0, 0.50)
      (414, 115, 13, 8, 51.5, 1.62)
      (364, 114, 3, 1, 0.0, 3.00)
      (374, 112, 2, 2, 0.0, 1.00)
      (350, 112, 1, 1, 0.0, 1.00)
      (616, 111, 13, 2, 7.0, 6.50)
      (352, 111, 1, 1, 0.0, 1.00)
      (405, 110, 3, 2, 1.5, 1.50)
      (601, 108, 1, 2, 0.0, 0.50)
      (378, 108, 26, 8, 110.0, 3.25)
      (0, 107, 374, 104, 10149.5, 3.60)
      (481, 106, 13, 24, 120.5, 0.54)
      (593, 105, 9, 19, 56.0, 0.47)
      (430, 105, 4, 2, 3.0, 2.00)
      (486, 103, 1, 2, 0.0, 0.50)
      (446, 103, 1, 1, 0.0, 1.00)
      (443, 103, 1, 1, 0.0, 1.00)
      (433, 103, 1, 1, 0.0, 1.00)
      (446, 101, 1, 1, 0.0, 1.00)
      (424, 100, 1, 1, 0.0, 1.00)
      (436, 99, 2, 5, 0.0, 0.40)
      (355, 99, 10, 8, 25.5, 1.25)
      (355, 99, 1, 1, 0.0, 1.00)
      (433, 98, 1, 1, 0.0, 1.00)
      (623, 97, 4, 2, 2.0, 2.00)
      (612, 95, 8, 4, 12.0, 2.00)
      (436, 95, 2, 1, 0.0, 2.00)
      (429, 94, 3, 4, 5.0, 0.75)
      (414, 94, 8, 8, 18.5, 1.00)
      (346, 93, 1, 1, 0.0, 1.00)
      (439, 92, 5, 12, 19.5, 0.42)
      (477, 91, 2, 2, 0.0, 1.00)
      (486, 88, 1, 1, 0.0, 1.00)
      (348, 88, 5, 3, 7.5, 1.67)
      (425, 86, 1, 2, 0.0, 0.50)
      (424, 83, 1, 2, 0.0, 0.50)
      (386, 80, 1, 1, 0.0, 1.00)
      (353, 80, 7, 4, 1.0, 1.75)
      (638, 78, 2, 1, 0.0, 2.00)
      (411, 78, 1, 1, 0.0, 1.00)
      (478, 77, 2, 2, 0.5, 1.00)
      (378, 77, 2, 1, 0.0, 2.00)
      (336, 77, 1, 1, 0.0, 1.00)
      (639, 76, 1, 1, 0.0, 1.00)
      (608, 76, 1, 1, 0.0, 1.00)
      (510, 76, 1, 1, 0.0, 1.00)
      (442, 76, 4, 7, 1.5, 0.57)
      (517, 75, 2, 2, 0.5, 1.00)
      (423, 75, 2, 2, 0.5, 1.00)
      (613, 74, 3, 6, 7.5, 0.50)
      (520, 74, 1, 2, 0.0, 0.50)
      (599, 73, 6, 6, 7.5, 1.00)
      (435, 73, 1, 1, 0.0, 1.00)
      (426, 73, 7, 3, 9.5, 2.33)
      (23, 73, 1, 1, 0.0, 1.00)
      (634, 72, 2, 7, 5.5, 0.29)
      (426, 72, 19, 17, 58.5, 1.12)
      (0, 72, 8, 14, 41.0, 0.57)
      (430, 71, 1, 1, 0.0, 1.00)
      (24, 71, 1, 1, 0.0, 1.00)
      (604, 70, 28, 10, 80.5, 2.80)
      (413, 70, 1, 1, 0.0, 1.00)
      (554, 67, 4, 2, 0.0, 2.00)
      (443, 66, 2, 1, 0.0, 2.00)
      (447, 65, 1, 1, 0.0, 1.00)
      (445, 63, 1, 2, 0.0, 0.50)
      (639, 62, 1, 1, 0.0, 1.00)
      (590, 62, 1, 1, 0.0, 1.00)
      (488, 62, 1, 1, 0.0, 1.00)
      (486, 62, 1, 1, 0.0, 1.00)
      (567, 61, 3, 3, 2.0, 1.00)
      (491, 61, 1, 1, 0.0, 1.00)
      (418, 61, 1, 1, 0.0, 1.00)
      (506, 60, 3, 1, 0.0, 3.00)
      (588, 59, 1, 1, 0.0, 1.00)
      (591, 58, 1, 2, 0.0, 0.50)
      (483, 57, 4, 2, 3.0, 2.00)
      (521, 56, 3, 5, 4.0, 0.60)
      (421, 56, 17, 11, 43.0, 1.55)
      (593, 55, 20, 6, 37.5, 3.33)
      (586, 55, 1, 2, 0.0, 0.50)
      (556, 55, 10, 10, 31.0, 1.00)
      (541, 55, 14, 17, 86.5, 0.82)
      (563, 54, 1, 1, 0.0, 1.00)
      (0, 53, 640, 427, 181846.0, 1.50)
      (537, 53, 1, 1, 0.0, 1.00)
      (406, 52, 16, 16, 91.5, 1.00)
      (482, 51, 80, 26, 660.0, 3.08)
      (476, 50, 13, 7, 42.0, 1.86)
      (565, 49, 18, 6, 48.5, 3.00)
      (583, 46, 57, 7, 154.5, 8.14)
      (443, 46, 37, 39, 843.5, 0.95)
      (27, 46, 1, 1, 0.0, 1.00)
      (616, 45, 1, 1, 0.0, 1.00)
      (404, 44, 39, 21, 201.0, 1.86)
      (384, 43, 1, 1, 0.0, 1.00)
      (422, 42, 11, 2, 8.5, 5.50)
      (241, 40, 1, 1, 0.0, 1.00)
      (365, 39, 13, 7, 24.0, 1.86)
      (49, 39, 1, 1, 0.0, 1.00)
      (411, 36, 23, 4, 30.5, 5.75)
      (19, 36, 5, 1, 0.0, 5.00)
      (15, 36, 1, 1, 0.0, 1.00)
      (617, 34, 23, 4, 31.5, 5.75)
      (304, 30, 1, 2, 0.0, 0.50)
      (64, 25, 4, 1, 0.0, 4.00)
      (488, 24, 152, 34, 876.5, 4.47)
      (56, 24, 4, 2, 0.0, 2.00)
      (214, 23, 1, 1, 0.0, 1.00)
      (424, 20, 1, 1, 0.0, 1.00)
      (431, 19, 1, 1, 0.0, 1.00)
      (106, 19, 4, 1, 0.0, 4.00)
      (482, 18, 1, 1, 0.0, 1.00)
      (210, 18, 2, 3, 1.0, 0.67)
      (208, 18, 1, 1, 0.0, 1.00)
      (476, 17, 4, 2, 2.0, 2.00)
      (251, 17, 4, 3, 4.5, 1.33)
      (101, 17, 3, 2, 0.5, 1.50)
      (487, 16, 6, 2, 1.0, 3.00)
      (467, 16, 5, 3, 3.5, 1.67)
      (0, 16, 206, 28, 834.0, 7.36)
      (335, 15, 1, 1, 0.0, 1.00)
      (0, 15, 448, 131, 36189.0, 3.42)
      (184, 15, 1, 1, 0.0, 1.00)
      (63, 15, 1, 1, 0.0, 1.00)
      (202, 14, 1, 1, 0.0, 1.00)
      (199, 14, 2, 1, 0.0, 2.00)
      (164, 14, 2, 1, 0.0, 2.00)
      (32, 13, 1, 1, 0.0, 1.00)
      (617, 11, 1, 1, 0.0, 1.00)
      (315, 8, 9, 5, 21.0, 1.80)
      (336, 7, 3, 1, 0.0, 3.00)
      (354, 5, 19, 2, 5.0, 9.50)
      (211, 5, 3, 2, 1.5, 1.50)
      (237, 4, 3, 1, 0.0, 3.00)
      (232, 4, 1, 1, 0.0, 1.00)
      (385, 3, 1, 1, 0.0, 1.00)
      (206, 3, 106, 16, 899.0, 6.62)
      (228, 2, 2, 1, 0.0, 2.00)
      (327, 1, 313, 49, 5369.5, 6.39)
      (276, 1, 1, 1, 0.0, 1.00)
      (459, 0, 17, 1, 0.0, 17.00)
      (315, 0, 142, 5, 247.5, 28.40)
      (308, 0, 6, 2, 3.0, 3.00)
      (305, 0, 1, 1, 0.0, 1.00)
      (272, 0, 1, 1, 0.0, 1.00)
      (246, 0, 6, 2, 1.0, 3.00)
      (244, 0, 1, 1, 0.0, 1.00)
      (235, 0, 5, 2, 4.0, 2.50)
      (222, 0, 2, 1, 0.0, 2.00)
      (209, 0, 11, 3, 9.0, 3.67)
      (207, 0, 1, 1, 0.0, 1.00)
      (173, 0, 33, 14, 335.0, 2.36)
      (0, 0, 172, 29, 2623.0, 5.93)



    
![png](anpr_plate_detection_files/anpr_plate_detection_7_1.png)
    


    Number of geometric candidates: 11


## 5. Edge Density and Texture Analysis

Analyze edge density and texture in candidate regions to further validate likely license plates.


```python
# Analyze edge density for each candidate
edge_density_threshold = (0.1, 0.8)  # Acceptable range for plates
final_candidates = []

for (x, y, w, h, aspect_ratio, area, contour) in candidates:
    roi = preprocessed[y:y+h, x:x+w]
    edges = cv2.Canny(roi, 50, 150)
    edge_pixels = np.sum(edges > 0)
    total_pixels = roi.shape[0] * roi.shape[1]
    density = edge_pixels / total_pixels if total_pixels > 0 else 0
    if edge_density_threshold[0] < density < edge_density_threshold[1]:
        final_candidates.append((x, y, w, h, density))

# Draw final candidates
final_img = image_rgb.copy()
for (x, y, w, h, density) in final_candidates:
    cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(final_img, f"{density:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

plt.figure(figsize=(8, 6))
plt.imshow(final_img)
plt.title('Final Plate Candidates (Edge Density)')
plt.axis('off')
plt.show()

print(f"Number of final candidates after edge density: {len(final_candidates)}")
```


    
![png](anpr_plate_detection_files/anpr_plate_detection_9_0.png)
    


    Number of final candidates after edge density: 11


## 6. Visualize Detected Plate Candidates

Display the final detected plate candidates on the original image for review.


```python
# The final_img with green boxes already shows the detected candidates.
# Optionally, display again for clarity.
plt.figure(figsize=(8, 6))
plt.imshow(final_img)
plt.title('Final Detected Plate Candidates')
plt.axis('off')
plt.show()
```


    
![png](anpr_plate_detection_files/anpr_plate_detection_11_0.png)
    


## 7. (Optional) Benchmarking and Performance Analysis

You can extend this notebook with benchmarking code to evaluate detection accuracy and speed, as described in the article.

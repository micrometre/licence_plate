import cv2
import numpy as np

# Load image
image = cv2.imread("./sample_images/1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize MSER detector
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(gray)

# Create a mask for MSER regions
mask = np.zeros_like(gray)
for region in regions:
    x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
    aspect_ratio = w / h
    if 1.5 < aspect_ratio < 6.0 and 30 < w < 300 and 10 < h < 100:  # Adjust based on plate size
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display result
cv2.imshow("MSER Plate Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
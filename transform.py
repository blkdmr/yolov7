import cv2

# Load image
image = cv2.imread('leaf2.jpeg')

# Convert to HSV to adjust saturation and hue
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Increase saturation and fine-tune hue
hsv[..., 1] = cv2.multiply(hsv[..., 1], 1.4)  # Saturation enhancement
hsv[..., 0] = cv2.add(hsv[..., 0], 2)         # Hue shift

# Convert back to BGR
image_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Apply CLAHE to each channel for local contrast enhancement
lab = cv2.cvtColor(image_sat, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)
lab_clahe = cv2.merge((l_clahe, a, b))
image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Apply unsharp masking to enhance edges and veins
blurred = cv2.GaussianBlur(image_clahe, (0, 0), sigmaX=1.5)
sharpened = cv2.addWeighted(image_clahe, 1.5, blurred, -0.5, 0)

cv2.imwrite('leaf_enhanced-2.jpg', sharpened)
# cv2.imshow('Enhanced Leaf', sharpened)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

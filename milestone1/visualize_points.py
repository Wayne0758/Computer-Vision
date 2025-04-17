import cv2

# Load the image
image = cv2.imread('G:\Machine Learning\CV\IMG_1226.jpg')
image = cv2.resize(image, (800, 600))  # Resize to fit screen

# Example list of points [(x1, y1), (x2, y2), ...]
points = [(192, 185),
(609, 233),
(624, 470),
(182, 513),
(280, 334),
(328, 335),
(583, 341),
(615, 342),
(365, 315),
(501, 321),
(508, 482),
(366, 494)]
# Draw all points
for idx, (x, y) in enumerate(points):
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red dot
    cv2.putText(image, f'{idx}: ({x}, {y})', (x + 10, y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(255, 255, 255), thickness=1)

# Show the image
cv2.imshow('Image with Points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

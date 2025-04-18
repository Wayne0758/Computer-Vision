import cv2

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")

image = cv2.imread('milestone1\Data\IMG_1226.JPG')
resized = cv2.resize(image, (800, 600))  # Resize to fit screen
cv2.imshow('Resized Image', resized)
cv2.setMouseCallback('Resized Image', mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()

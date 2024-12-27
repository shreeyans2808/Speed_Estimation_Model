import cv2
   
points = []
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
image = cv2.imread('/Users/shreeyansarora/Downloads/Internship_Work/Task5/frame_0.jpg')
cv2.imshow('Image', image)
cv2.setMouseCallback('Image', select_points)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(points)

import cv2
from  random import randrange

trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
test_image = cv2.imread('group-of-young-children-hanging-out-in-park-picture-id495354633.jpg')
gray_scaled_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
face_coordinates = trained_data.detectMultiScale(gray_scaled_test_image)
for (x_axis, y_axis, width, height) in face_coordinates:
    cv2.rectangle(test_image, (x_axis, y_axis), (x_axis+width, y_axis+height), (randrange(256), randrange(256), randrange(256)), 5)

cv2.imshow("test face detector", test_image)
cv2.waitKey()
print("complete")
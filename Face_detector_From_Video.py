import cv2
from  random import randrange

trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
test_video = cv2.VideoCapture('test_video.mp4')

while True:
    successful_frame_read, frame_image = test_video.read()
    gray_scaled_test_video_frame = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_data.detectMultiScale(gray_scaled_test_video_frame)

    for (x_axis, y_axis, width, height) in face_coordinates:
        cv2.rectangle(frame_image, (x_axis, y_axis), (x_axis+width, y_axis+height), (randrange(256), randrange(256), randrange(256)), 5)

    cv2.imshow("test face detector", frame_image)
    key = cv2.waitKey(1)

    if key==81 or key == 113:
        break


print("complete")

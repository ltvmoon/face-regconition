import cv2
import numpy as np

# Chuẩn bị dữ liệu
face_cascade = cv2.CascadeClassifier("database/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("database/haarcascade_eye.xml")
nose_cascade = cv2.CascadeClassifier("database/haarcascade_nose.xml")
mouth_cascade = cv2.CascadeClassifier("database/haarcascade_mouth.xml")

# Tải hình ảnh
image = cv2.imread("data/image.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Phát hiện khuôn mặt
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Vẽ khung xung quanh các khuôn mặt
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Phát hiện mắt
eyes = eye_cascade.detectMultiScale(gray, 3, 5)

# Vẽ khung xung quanh các mắt
for (x, y, w, h) in eyes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Phát hiện mũi
nose = nose_cascade.detectMultiScale(gray, 3, 5)

# Vẽ khung xung quanh mũi
for (x, y, w, h) in nose:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Phát hiện miệng
mouths = mouth_cascade.detectMultiScale(gray, 3, 5)

# Vẽ khung xung quanh miệng
for (x, y, w, h) in mouths:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

# Hiển thị hình ảnh
cv2.imshow("Image", image)
cv2.waitKey(0)
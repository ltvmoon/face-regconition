import cv2
import numpy as np
from PIL import Image


def white_balance_RGB(image):
    # Lấy các giá trị RGB trung bình của hình ảnh.
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]
    r_mean = red_channel.mean() / 255
    g_mean = green_channel.mean() / 255
    b_mean = blue_channel.mean() / 255

    # Tạo một ma trận cân bằng trắng.
    white_balance_matrix = np.array([[r_mean, 0, 0],
                                     [0, g_mean, 0],
                                     [0, 0, b_mean]], dtype=np.float32)
    # Cân bằng trắng hình ảnh.
    image_array = np.array(image)
    balanced_image_array = np.dot(image_array, white_balance_matrix)
    balanced_image_array = np.clip(balanced_image_array, 0, 255).astype(np.uint8)
    white_balanced_image = Image.fromarray(balanced_image_array)
    return white_balanced_image


# Cân bằng trắng dựa trên thuật toán của thế giới xám
# Chúng ta sẽ chuyển đổi hình ảnh sang không gian màu LAB: L cho độ sáng, A cho Đỏ/Xanh lục và B cho Xanh lam/Vàng
# Chúng ta sẽ tính toán giá trị màu trung bình trong kênh A và B.
# Sau đó, trừ 128 (màu xám giữa) khỏi giá trị trung bình và chuẩn hóa kênh L bằng cách nhân với sự khác biệt này.
# Cuối cùng, trừ giá trị này khỏi kênh A và B.
# Bạn có thể thêm hệ số nhân để tăng/giảm độ sáng tổng thể
# của mỗi kênh A hoặc B. (Ở đây tôi đã thêm 1,2 làm hệ số nhân)
def LAB_white_balance(img):
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img_LAB[:, :, 1])
    avg_b = np.average(img_LAB[:, :, 2])
    img_LAB[:, :, 1] = img_LAB[:, :, 1] - ((avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    img_LAB[:, :, 2] = img_LAB[:, :, 2] - ((avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    balanced_image = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
    return balanced_image


def noise_reduction_Gauss(path):
    image = cv2.imread(path, 0)

    # Tạo kernel làm mờ 3x3
    kernel = np.ones((3, 3), np.float32) / 6
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    # Áp dụng bộ lọc lên ảnh
    filtered_image = cv2.filter2D(blurred_image, -1, kernel)
    return filtered_image


def Contrast_enhancement(path):
    image = cv2.imread(path, 0)  # Load the image in grayscale
    equalized_image = cv2.equalizeHist(image)
    return equalized_image


def Resize_image_800_600(image):
    scale = min(float(800 / image.shape[1]), float(600 / image.shape[0]))
    scaled_image = cv2.resize(image, None, fx=scale, fy=scale)
    return scaled_image


def face_recognize(image):
    face_cascade = cv2.CascadeClassifier("database/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("database/haarcascade_eye.xml")
    faces = face_cascade.detectMultiScale(image, 1.1, 6)
    face_pos = []
    # Vẽ khung xung quanh các khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_pos.append([x, y, w, h])

    # Phát hiện mắt
    eyes = eye_cascade.detectMultiScale(image, 1.1, 5)

    # Vẽ khung xung quanh các mắt
    for (x1, y1, w1, h1) in face_pos:
        for (x, y, w, h) in eyes:
            if (x > x1) & (x < x1 + w1) & (y > y1) & (y < y1 + h1) & (x + w < x1 + w1) & (y + h < y1 + h1):
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                continue

    return image

import cv2
import numpy as np
from PIL import Image

def white_balance(image):
  """
  Cân bằng trắng cho hình ảnh.

  Args:
    image: Hình ảnh đầu vào.

  Returns:
    Hình ảnh đã cân bằng trắng.
    :param image_path:
  """

  # Lấy các giá trị RGB trung bình của hình ảnh.
  red_channel = image[:, :, 2]
  green_channel = image[:, :, 1]
  blue_channel = image[:, :, 0]
  r_mean = red_channel.mean() / 255
  g_mean = green_channel.mean() /255
  b_mean = blue_channel.mean() /255

  # Tạo một ma trận cân bằng trắng.
  white_balance_matrix = np.array([[r_mean, 0, 0],
                                   [0, g_mean , 0],
                                   [0, 0, b_mean]],dtype=np.float32)
  # Cân bằng trắng hình ảnh.
  image_array = np.array(image)
  balanced_image_array = np.dot(image_array, white_balance_matrix)
  balanced_image_array = np.clip(balanced_image_array, 0, 255).astype(np.uint8)
  white_balanced_image = Image.fromarray(balanced_image_array)
  return white_balanced_image

def noise_reduction_Gauss():
  image = cv2.imread("result/white_balanced_image.jpg",0)

  # Tạo kernel làm mờ 3x3
  kernel = np.ones((4, 4), np.float32) / 6
  blurred_image = cv2.GaussianBlur(image, (5, 5), -1)
  # Áp dụng bộ lọc lên ảnh
  filtered_image = cv2.filter2D(blurred_image, -1, kernel)
  cv2.imwrite("result/filtered_image.jpg", filtered_image)
  pass
def Contrast_enhancement():
  image = cv2.imread("result/filtered_image.jpg", 0)  # Load the image in grayscale
  equalized_image = cv2.equalizeHist(image)
  cv2.imwrite("result/equalized_image.jpg", equalized_image)
  pass
import cv2
import Image_preprocessing as Ip
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import sys
import subprocess


def main():
  """
  Chạy chương trình.
  """

  # Đọc hình ảnh.
  image_path = "data/image.jpg"
  image = cv2.imread(image_path)
  # Cân bằng trắng hình ảnh.
  white_balanced_image = Ip.white_balance(image)
  white_balanced_image.save('result/white_balanced_image.jpg')
  white_balanced_image.close()
  # Lọc nhiễu gauss _ chuyển sang dạng xám
  Ip.noise_reduction_Gauss()

  # Tăng cường tương phản bằng histogram
  Ip.Contrast_enhancement()

  # hiển thị kết quả
  subprocess.run(["python", "display.py"])



if __name__ == "__main__":
  main()
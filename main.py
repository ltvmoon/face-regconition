import cv2
import Image_preprocessing as Ip
import subprocess
import tkinter as tk
import tkinter.filedialog

Pathuse =""
def get_path(window):
  # Mở hộp thoại cho phép người dùng chọn file hình ảnh
  filename = tk.filedialog.askopenfilename(
    initialdir="/",
    filetypes=[("Ảnh", "*.jpg")]
  )
  global Pathuse
  Pathuse = filename
  window.destroy()
  return filename
def main():
  """
  Chạy chương trình.
  """

  window = tk.Tk()
  window.geometry("200x100+100+100")
  # Tạo một nút để chọn file hình ảnh
  button = tk.Button(
    window,
    text="Chọn file hình ảnh",
    command=lambda: get_path(window)
  )
  button.pack()
  window.mainloop()


  # Đọc hình ảnh.
  image_path = Pathuse
  original_image = cv2.imread(image_path)
  image = Ip.Resize_image_800_600(original_image)
  cv2.imwrite("data/scaled_image.jpg", image)

  # Cân bằng trắng hình ảnh bằng trung bình RGB
  white_balanced_image_RGB = Ip.white_balance_RGB(original_image)
  white_balanced_image_RGB.save('result/white_balanced_image_RGB.jpg')
  white_balanced_image_RGB.close()
  white_balanced_image_RBG = cv2.imread('result/white_balanced_image_RGB.jpg')
  image = Ip.Resize_image_800_600(white_balanced_image_RBG)
  cv2.imwrite("result/scaled_white_balanced_image_RGB.jpg", image)

  # Cân bằng trắng hình ảnh bằng trung bình LAB
  white_balanced_image_LAB = Ip.LAB_white_balance(original_image)
  cv2.imwrite("result/white_balanced_image_LAB.jpg", white_balanced_image_LAB)
  image = Ip.Resize_image_800_600(white_balanced_image_LAB)
  cv2.imwrite("result/scaled_white_balanced_image_LAB.jpg", image)

  # Lọc nhiễu gauss _ chuyển sang dạng xám
  Gauss_RGB = Ip.noise_reduction_Gauss("result/white_balanced_image_RGB.jpg")
  cv2.imwrite("result/Gauss_image_RGB.jpg", Gauss_RGB)
  image = Ip.Resize_image_800_600(Gauss_RGB)
  cv2.imwrite("result/scaled_Gauss_image_RGB.jpg", image)
  Gauss_lAB = Ip.noise_reduction_Gauss("result/white_balanced_image_LAB.jpg")
  cv2.imwrite("result/Gauss_image_LAB.jpg", Gauss_lAB)
  image = Ip.Resize_image_800_600(Gauss_lAB)
  cv2.imwrite("result/scaled_Gauss_image_LAB.jpg", image)

  # Tăng cường tương phản bằng histogram
  Contrast1= Ip.Contrast_enhancement("result/Gauss_image_RGB.jpg")
  cv2.imwrite("result/Contrast_en_image_RGB.jpg", Contrast1)
  image = Ip.Resize_image_800_600(Contrast1)
  cv2.imwrite("result/scaled_Contrast_en_image_RGB.jpg", image)
  Contrast2 = Ip.Contrast_enhancement("result/Gauss_image_LAB.jpg")
  cv2.imwrite("result/Contrast_en_image_LAB.jpg", Contrast2)
  image = Ip.Resize_image_800_600(Contrast2)
  cv2.imwrite("result/scaled_Contrast_en_image_LAB.jpg", image)

  RGB1 = cv2.imread("result/Contrast_en_image_RGB.jpg")
  RGB_reg = Ip.face_recognize(RGB1)
  cv2.imwrite("result/regconized_face_image_LAB.jpg", RGB_reg)
  image = Ip.Resize_image_800_600(RGB_reg)
  cv2.imwrite("result/scaled_regconized_face_image_LAB.jpg", image)

  LAB1 = cv2.imread("result/Contrast_en_image_LAB.jpg")
  LAB_reg = Ip.face_recognize(LAB1)
  cv2.imwrite("result/regconized_face_image_RGB.jpg", LAB_reg)
  image = Ip.Resize_image_800_600(LAB_reg)
  cv2.imwrite("result/scaled_regconized_face_image_RGB.jpg", image)

  # hiển thị kết quả
  subprocess.run(["python", "display.py"])



if __name__ == "__main__":
  main()
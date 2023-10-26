import tkinter as tk
from PIL import ImageTk, Image

# Create the Tkinter window
window = tk.Tk()

# Create a list of image paths and titles
image_data = [
    {"path": "data/scaled_image.jpg", "title": "step 1 : Load Original_image"},
    {"path": "result/scaled_white_balanced_image_RGB.jpg", "title": "step 2: White_balanced_image_with_RGB"},
    {"path": "result/scaled_white_balanced_image_LAB.jpg", "title": "step 2: White_balanced_image_with_LAB"},
    {"path": "result/scaled_Gauss_image_RGB.jpg", "title": "step 3: Gauss_image_with_RGB"},
    {"path": "result/scaled_Gauss_image_LAB.jpg", "title": "step 3: Gauss_image_with_LAB"},
    {"path": "result/scaled_Contrast_en_image_RGB.jpg", "title": "step 4: Contrast_en_image_RGB"},
    {"path": "result/scaled_Contrast_en_image_LAB.jpg", "title": "step 4 : Contrast_en_image_LAB"},
    {"path": "result/scaled_regconized_face_image_RGB.jpg", "title": "step 5 : Regconized_face_image_RGB"},
    {"path": "result/scaled_regconized_face_image_LAB.jpg", "title": "step 5 : Regconized_face_image_LAB"}
]

# Create variables to keep track of the current image index and the image label
current_image_index = 0
image_label = None
title_label = None

# Function to create and display the images
def create_image():
    global current_image_index, image_label, title_label

    # Clear the previous image and title labels if they exist
    if image_label:
        image_label.destroy()
    if title_label:
        title_label.destroy()

    # Check if we have reached the end of the image list
    if current_image_index >= len(image_data):
        current_image_index = len(image_data) -1

    # Load the image and create a Tkinter PhotoImage object
    image = Image.open(image_data[current_image_index]["path"])
    image_tk = ImageTk.PhotoImage(image)

    # Create a label widget to display the image
    image_label = tk.Label(window, image=image_tk)
    image_label.image = image_tk  # Keep a reference to prevent garbage collection

    # Position the image label in the first column
    image_label.grid(row=2, column=0, padx=10, pady=10)

    # Create a label widget to display the title
    title_label = tk.Label(window, text=image_data[current_image_index]["title"])
    title_label.grid(row=3, column=0)

    # Increment the current image index
    current_image_index += 1

# Function to handle the button click event
def next_image():
    create_image()

# Create the button
button = tk.Button(window, text="Next", command=next_image)
button.grid(row=0, column=0, padx=10, pady=10)
quit_button = tk.Button(window, text="Quit", command=window.quit)
quit_button.grid(row=1, column=0, padx=10, pady=10)


# Call the create_image function to display the first image
create_image()

# Start the Tkinter event loop
window.mainloop()
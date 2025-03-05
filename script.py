import os
import csv
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd

# Folder with images
folder_path = filedialog.askdirectory(title='Select the folder with images')
images = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# Create CSV file if it does not exist
csv_file = 'classifications.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Asymmetry', 'Border', 'Color'])

# Read CSV correctly
data = pd.read_csv(csv_file) if os.path.exists(csv_file) and os.stat(csv_file).st_size > 0 else pd.DataFrame(columns=['Image', 'Asymmetry', 'Border', 'Color'])
classified_images = set(data['Image']) if not data.empty else set()

# Filter unclassified images
images = [img for img in images if img not in classified_images]
current_index = 0
selected_values = {'Asymmetry': None, 'Border': None, 'Color': None}
buttons = {'Asymmetry': {}, 'Border': {}, 'Color': {}}

# Function to display the image
def show_image():
    global current_index, img_label, img, selected_values
    if current_index < len(images):
        img_path = os.path.join(folder_path, images[current_index])
        image = Image.open(img_path)
        image = image.resize((500, 500), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(image)
        img_label.config(image=img)
        img_label.image = img
        status_label.config(text=f'Image {current_index + 1} of {len(images)}')
        selected_values = {'Asymmetry': None, 'Border': None, 'Color': None}  # Reset selections
        reset_button_styles()
    else:
        status_label.config(text='Classification completed!')
        root.quit()

# Function to check if all criteria are selected
def all_criteria_selected():
    return all(value is not None for value in selected_values.values())

# Function to save the results
def save_classification():
    global current_index, data
    if current_index < len(images) and all_criteria_selected():
        img_name = images[current_index]
        new_data = pd.DataFrame([[img_name, selected_values['Asymmetry'], selected_values['Border'], selected_values['Color']]],
                                columns=['Image', 'Asymmetry', 'Border', 'Color'])
        data = pd.concat([data, new_data], ignore_index=True)
        data.to_csv(csv_file, index=False)
        current_index += 1
        show_image()

# Function to set classification and highlight the selected button
def set_classification(category, value):
    selected_values[category] = value
    update_button_styles(category, value)
    if all_criteria_selected():
        save_classification()

# Function to update button styles
def update_button_styles(category, selected_value):
    for value, button in buttons[category].items():
        if value == selected_value:
            button.config(relief=tk.SUNKEN, bg='lightblue')
        else:
            button.config(relief=tk.RAISED, bg=root.cget("bg"))

# Function to reset button styles when a new image is displayed
def reset_button_styles():
    for category in buttons:
        for button in buttons[category].values():
            button.config(relief=tk.RAISED, bg=root.cget("bg"))

# Create interface
root = tk.Tk()
root.title('Dermatological Image Classification')

img_label = tk.Label(root)
img_label.pack()

status_label = tk.Label(root, text='Starting...', font=('Arial', 12))
status_label.pack()

frame = tk.Frame(root)
frame.pack()

criteria = {'Asymmetry': [0, 1], 'Border': [0, 1], 'Color': [0, 1]}

def create_buttons():
    for category, values in criteria.items():
        for value in values:
            btn = tk.Button(frame, text=f'{category}: {value}', command=lambda c=category, v=value: set_classification(c, v))
            btn.pack(side=tk.LEFT, padx=5)
            buttons[category][value] = btn

create_buttons()
show_image()
root.mainloop()

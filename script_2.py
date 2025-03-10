import os
import csv
import tkinter as tk
from tkinter import filedialog, ttk, StringVar
from PIL import Image, ImageTk
import pandas as pd

# Folder with images
folder_path = filedialog.askdirectory(title='Select the folder with images')
images = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# Create CSV file if it does not exist
csv_file = 'classifications__3.csv'
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
selected_values = {'Asymmetry': None, 'Border': None, 'Color': []}

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
        selected_values['Asymmetry'] = None
        selected_values['Border'] = None
        selected_values['Color'] = []
        asymmetry_var.set('Select')
        border_var.set('Select')
        for var in color_vars.values():
            var.set(0)
    else:
        status_label.config(text='Classification completed!')
        root.quit()

# Function to check if all criteria are selected
def all_criteria_selected():
    return selected_values['Asymmetry'] is not None and selected_values['Asymmetry'] != 'Select' \
           and selected_values['Border'] is not None and selected_values['Border'] != 'Select' \
           and selected_values['Color']

# Function to save the results
def save_classification():
    global current_index, data
    if current_index < len(images) and all_criteria_selected():
        img_name = images[current_index]
        colors_selected = ', '.join(selected_values['Color'])
        new_data = pd.DataFrame([[img_name, selected_values['Asymmetry'], selected_values['Border'], colors_selected]],
                                columns=['Image', 'Asymmetry', 'Border', 'Color'])
        data = pd.concat([data, new_data], ignore_index=True)
        data.to_csv(csv_file, index=False)
        current_index += 1
        show_image()

# Function to set classification
def set_classification(category, value):
    if category == 'Color':
        if value in selected_values['Color']:
            selected_values['Color'].remove(value)
        else:
            selected_values['Color'].append(value)
    else:
        selected_values[category] = value
    if all_criteria_selected():
        save_classification()

# Create interface
root = tk.Tk()
root.title('Dermatological Image Classification')

img_label = tk.Label(root)
img_label.pack()

status_label = tk.Label(root, text='Starting...', font=('Arial', 12))
status_label.pack()

frame = tk.Frame(root)
frame.pack()

criteria = {
    'Asymmetry': ["Select", 0, 1, 2],
    'Border': ["Select", 0, 1, 2],
    'Color': ['white', 'red', 'light-brown', 'dark-brown', 'blue-gray', 'black']
}

asymmetry_var = tk.StringVar(value='Select')
border_var = tk.StringVar(value='Select')

# Dropdown for Asymmetry
ttk.Label(frame, text='Asymmetry:').pack(side=tk.LEFT, padx=5)
asymmetry_dropdown = ttk.Combobox(frame, textvariable=asymmetry_var, values=criteria['Asymmetry'])
asymmetry_dropdown.pack(side=tk.LEFT, padx=5)
asymmetry_dropdown.bind("<<ComboboxSelected>>", lambda e: set_classification('Asymmetry', asymmetry_var.get()))

# Dropdown for Border
ttk.Label(frame, text='Border:').pack(side=tk.LEFT, padx=5)
border_dropdown = ttk.Combobox(frame, textvariable=border_var, values=criteria['Border'])
border_dropdown.pack(side=tk.LEFT, padx=5)
border_dropdown.bind("<<ComboboxSelected>>", lambda e: set_classification('Border', border_var.get()))

# Checkboxes for Color
ttk.Label(frame, text='Color:').pack(side=tk.LEFT, padx=5)
color_vars = {}
color_frame = tk.Frame(root)
color_frame.pack()

for color in criteria['Color']:
    var = tk.IntVar(value=0)
    chk = tk.Checkbutton(color_frame, text=color, variable=var, 
                          command=lambda c=color, v=var: set_classification('Color', c if v.get() else None))
    chk.pack(side=tk.LEFT, padx=2)
    color_vars[color] = var

show_image()
root.mainloop()
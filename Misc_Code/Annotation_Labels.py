import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import csv

FOLDER = r"C:/Users/Helene/Desktop/Uni/Masterarbeit/Dataset/trash_neu"
CSV_FILE = "annotation_new_photos.csv"
START_NUMBER = 5132

LABELS = [
    "Aluminium",
    "Plastic",
    "Metal",
    "Residual waste",
    "Cardboard",
    "Organic waste",
    "Composite Carton",
    "Paper",
    "Brown Glass",
    "Plastic,Aluminium",
    "Green Glass",
    "White Glass",
    "White Glass,Metal",
    "PET",
    "Rigid plastic container",
    "Hazardous waste (Battery)",
    "Discard"
]

class Annotator:
    def __init__(self, master, image_folder, csv_file, labels, start_number):
        self.master = master
        self.image_folder = image_folder
        self.csv_file = csv_file
        self.labels = labels
        self.current_number = start_number

        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.index = 0
        self.results = []

        self.img_label = tk.Label(master)
        self.img_label.pack()

        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack(pady=10)

        self.buttons = []
        columns = 4  # Buttons per row
        for i, lbl in enumerate(labels):
            btn = tk.Button(self.btn_frame, text=lbl, command=lambda l=lbl: self.annotate(l), width=18, height=2)
            btn.grid(row=i // columns, column=i % columns, padx=4, pady=4)
            self.buttons.append(btn)

        self.load_next_image()

    def load_next_image(self):
        if self.index >= len(self.image_files):
            messagebox.showinfo("Done", "All images have been annotated.")
            self.save_csv()
            self.master.quit()
            return
        image_path = os.path.join(self.image_folder, self.image_files[self.index])
        img = Image.open(image_path)
        img = img.resize((350, 350))
        self.tk_img = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_img)
        self.img_label.image = self.tk_img  # Keep a reference
        self.master.title(f"{self.index + 1}/{len(self.image_files)}: {self.image_files[self.index]}")

    def annotate(self, label):
        self.results.append([self.image_files[self.index], label, self.current_number])
        self.index += 1
        self.current_number += 1
        self.load_next_image()

    def save_csv(self):
        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "number"])
            writer.writerows(self.results)
        print(f"CSV saved as {self.csv_file}")

if __name__ == "__main__":
    root = tk.Tk()
    app = Annotator(root, FOLDER, CSV_FILE, LABELS, START_NUMBER)
    root.mainloop()

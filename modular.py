import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

CLASS_NAMES = ['Anthracnose', 'Helathy', 'Leaf Crinckle', 'Powdery Mildew', 'Yellow Mosaic']
def dummy_predict(image_path):
    label = hash(image_path) % len(CLASS_NAMES)
    return CLASS_NAMES[label]

def dummy_ground_truth(image_path):
    for class_name in CLASS_NAMES:
        if class_name.lower() in image_path.lower():
            return class_name
    return CLASS_NAMES[0]

class ModelPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Predictor")
        self.model = None
        self.image_paths = []

        self.select_model_btn = tk.Button(root, text="Select Model", command=self.select_model)
        self.select_model_btn.pack(pady=5)

        self.select_image_btn = tk.Button(root, text="Select Image(s)", command=self.select_images)
        self.select_image_btn.pack(pady=5)

        self.predict_btn = tk.Button(root, text="Predict", command=self.predict)
        self.predict_btn.pack(pady=5)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

    def select_model(self):
        self.model = filedialog.askopenfilename(title="Select Model")
        print("Model selected:", self.model)

    def select_images(self):
        selection_type = messagebox.askyesno("Selection Type", 
                                            "Do you want to select a single image?\n"
                                            "Select 'Yes' for single image, 'No' for folder")
        
        if selection_type:
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
            )
            if file_path:
                self.image_paths = [file_path]
                print("Image loaded:", self.image_paths)
                self.display_image(self.image_paths[0])
        else:
            folder_path = filedialog.askdirectory(title="Select Image Folder")
            if folder_path:
                self.image_paths = [
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                print("Images loaded:", self.image_paths)
                if self.image_paths:
                    self.display_image(self.image_paths[0])

    def display_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((256, 256))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def predict(self):
        if not self.image_paths:
            messagebox.showerror("Error", "No images selected.")
            return

        y_true = []
        y_pred = []

        results = []
        for img_path in self.image_paths:
            try:
                true_label = dummy_ground_truth(img_path)
                pred_label = dummy_predict(img_path)
                y_true.append(true_label)
                y_pred.append(pred_label)
                results.append(f"{os.path.basename(img_path)}: {pred_label}")
            except Exception as e:
                print("Prediction error:", e)
                results.append(f"Error predicting {os.path.basename(img_path)}")

        messagebox.showinfo("Prediction Results", "\n".join(results))

        if len(self.image_paths) > 1:
            self.save_confusion_matrix(y_true, y_pred, multi_class=True)
        else:
            messagebox.showinfo("Single Image Prediction", 
                            f"True: {y_true[0]}\nPredicted: {y_pred[0]}")
            if not self.image_paths:
                messagebox.showerror("Error", "No images selected.")
                return

            y_true = []
            y_pred = []

            results = []
            for img_path in self.image_paths:
                try:
                    true_label = dummy_ground_truth(img_path)
                    pred_label = dummy_predict(img_path)
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                    results.append(f"{os.path.basename(img_path)}: {pred_label}")
                except Exception as e:
                    print("Prediction error:", e)
                    results.append(f"Error predicting {os.path.basename(img_path)}")

            messagebox.showinfo("Prediction Results", "\n".join(results))

            if len(self.image_paths) > 1:
                self.save_confusion_matrix(y_true, y_pred, multi_class=True)
            else:
                self.save_confusion_matrix(y_true, y_pred)

    def save_confusion_matrix(self, y_true, y_pred, save_path=None, multi_class=False):
        cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path)
        elif multi_class:
            import os
            accuracy = np.sum(np.diag(cm)) / np.sum(cm)
            accuracy_file = os.path.join(os.getcwd(), 'accuracy_matrix.txt')
            with open(accuracy_file, 'a') as f:
                f.write(f"Model accuracy: {accuracy:.4f}\n")
            plt.savefig(os.path.join(os.getcwd(), 'confusion_matrix.png'))
            plt.show()
        # else:
            # plt.show()
            
        plt.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelPredictorApp(root)
    root.mainloop()
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        self.root.geometry("800x600")

        self.video_stream = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.running = False

        # UI Components
        self.start_button = tk.Button(self.root, text="Start Detection", command=self.start_detection, bg="green", fg="white", font=("Arial", 12, "bold"))
        self.start_button.pack(pady=20)

        self.stop_button = tk.Button(self.root, text="Stop Detection", command=self.stop_detection, bg="red", fg="white", font=("Arial", 12, "bold"))
        self.stop_button.pack(pady=20)

        self.canvas = tk.Canvas(self.root, width=800, height=450, bg="black")
        self.canvas.pack()

    def start_detection(self):
        if not self.running:
            self.running = True
            self.video_stream = cv2.VideoCapture(0)  # Initialize webcam
            if not self.video_stream.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                self.running = False
                return

            self.update_frame()

    def stop_detection(self):
        self.running = False
        if self.video_stream:
            self.video_stream.release()
            self.video_stream = None

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.video_stream.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.5, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk  # Keep reference to avoid garbage collection

        self.root.after(10, self.update_frame)  # Schedule next frame update

    def on_close(self):
        self.stop_detection()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)  # Handle window close
    root.mainloop()

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import platform
import argparse

# --- Parse arguments ---
parser = argparse.ArgumentParser(description="YOLO + Tkinter Webcam App")
parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (.pt file)")
parser.add_argument("--conf", type=float, default=0.78,
                    help="Confidence threshold for filtering detections (default=0.78)")
args = parser.parse_args()

# --- Load YOLO model ---
model = YOLO(args.model)

# --- OpenCV webcam setup (cross-platform) ---
if "Windows" in platform.system():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0) 

# --- Tkinter setup ---
root = tk.Tk()
root.title("YOLO Webcam Detection")
root.geometry("800x600")

# Label to display video
video_label = tk.Label(root)
video_label.pack()

# Flag for detection control
running = True

def update_frame():
    if not running:
        root.after(10, update_frame)
        return

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    # Run YOLO inference with confidence threshold
    results = model(frame, conf=args.conf, verbose=False)

    # Draw detections
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < args.conf:
                continue  # skip low-confidence predictions

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            label_name = model.names[cls]

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # Convert for Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Refresh
    root.after(10, update_frame)

def toggle_detection():
    global running
    running = not running
    btn_toggle.configure(text="Start Detection" if not running else "Stop Detection")

def on_closing():
    global cap
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Button to start/stop detection
btn_toggle = tk.Button(root, text="Stop Detection", command=toggle_detection)
btn_toggle.pack()

# Handle window close
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start loop
update_frame()
root.mainloop()

"""
Object_detection Contact Tracing - Object Contamination Detection
Detects when persons touch bottles/cups and marks them contaminated.
UI: tkinter with live camera feed, stats panel, alert log and controls.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import queue
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk

# ─── Classes ────────────────────────────────────────────────────────────────
OBJECTS    = [39, 40, 41]
PERSONS    = [0]
CLASS_NAMES = {0: 'Person', 39: 'Bottle', 40: 'Glass', 41: 'Cup'}

# ─── Detection helpers ───────────────────────────────────────────────────────
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0: return 0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def get_palms(keypoints, shape):
    palms = []
    h, w = shape[:2]
    if keypoints is None or len(keypoints) < 11: return palms
    for elbow_i, wrist_i in [(7, 9), (8, 10)]:
        e, wr = keypoints[elbow_i], keypoints[wrist_i]
        if len(e) > 2 and e[2] > 0.25 and wr[2] > 0.25:
            if wr[0] > 0 and wr[1] > 0:
                px = int(wr[0] + 0.35*(wr[0]-e[0]))
                py = int(wr[1] + 0.35*(wr[1]-e[1]))
                palms.append({'center': (px, py),
                               'box': [max(0,px-55), max(0,py-55),
                                       min(w,px+55), min(h,py+55)]})
    return palms

# ─── App ─────────────────────────────────────────────────────────────────────
class ContactTracingApp:
    # colour palette
    BG       = "#1a1a2e"
    PANEL_BG = "#16213e"
    ACCENT   = "#0f3460"
    GREEN    = "#00d084"
    RED      = "#e94560"
    YELLOW   = "#f5a623"
    TEXT     = "#e0e0e0"
    SUBTEXT  = "#888888"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Object_detection Contact Tracing System")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        # State
        self.running       = False
        self.contaminated  = {}
        self.alert_queue   = queue.Queue()
        self.frame_queue   = queue.Queue(maxsize=2)
        self.camera_index  = tk.IntVar(value=0)
        self._thread       = None
        self._cap          = None
        self._models_ready = False
        self.yolo_model    = None
        self.pose_model    = None

        self._build_ui()
        self._poll()

    # ── UI Layout ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Title bar
        title_bar = tk.Frame(self.root, bg=self.ACCENT, pady=5)
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="  Object Detection Contact Tracing",
                 font=("Segoe UI", 11, "bold"),
                 bg=self.ACCENT, fg=self.TEXT).pack(side="left")
        tk.Label(title_bar, text="YOLOv8  |  Real-Time  ",
                 font=("Segoe UI", 8),
                 bg=self.ACCENT, fg=self.SUBTEXT).pack(side="right")

        # Body
        body = tk.Frame(self.root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=6, pady=4)

        # Left: camera feed
        left = tk.Frame(body, bg=self.PANEL_BG)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="LIVE FEED", font=("Segoe UI", 8, "bold"),
                 bg=self.PANEL_BG, fg=self.SUBTEXT).pack(pady=(6,0))
        # Placeholder black image so label starts at pixel size 480x360
        placeholder = ImageTk.PhotoImage(Image.new("RGB", (480, 360), (0, 0, 0)))
        self.video_label = tk.Label(left, image=placeholder, bg="black")
        self.video_label.image = placeholder
        self.video_label.pack(padx=4, pady=4)

        # Right: sidebar
        right = tk.Frame(body, bg=self.BG, width=210)
        right.pack(side="right", fill="y", padx=(6,0))
        right.pack_propagate(False)

        self._build_stats(right)
        self._build_controls(right)
        self._build_log(right)

        # Status bar
        self.status_var = tk.StringVar(value="Ready — press Start")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Segoe UI", 8), bg=self.ACCENT, fg=self.TEXT,
                 anchor="w", padx=8, pady=3).pack(fill="x", side="bottom")

    def _build_stats(self, parent):
        frame = tk.Frame(parent, bg=self.PANEL_BG)
        frame.pack(fill="x", pady=(0,8))
        tk.Label(frame, text="STATISTICS", font=("Segoe UI", 8, "bold"),
                 bg=self.PANEL_BG, fg=self.SUBTEXT).pack(pady=(8,4))

        row = tk.Frame(frame, bg=self.PANEL_BG)
        row.pack(fill="x", padx=8, pady=(0,8))

        def stat_card(parent, title, color):
            card = tk.Frame(parent, bg=self.ACCENT, padx=6, pady=6)
            lbl  = tk.Label(card, text="0", font=("Segoe UI", 16, "bold"),
                            bg=self.ACCENT, fg=color)
            lbl.pack()
            tk.Label(card, text=title, font=("Segoe UI", 6),
                     bg=self.ACCENT, fg=self.SUBTEXT).pack()
            return card, lbl

        p_card, self.lbl_persons      = stat_card(row, "PERSONS",      self.YELLOW)
        o_card, self.lbl_objects      = stat_card(row, "OBJECTS",      self.GREEN)
        c_card, self.lbl_contaminated = stat_card(row, "CONTAMINATED", self.RED)
        for card in (p_card, o_card, c_card):
            card.pack(side="left", expand=True, fill="x", padx=2)

    def _build_controls(self, parent):
        frame = tk.Frame(parent, bg=self.PANEL_BG)
        frame.pack(fill="x", pady=(0,8))
        tk.Label(frame, text="CONTROLS", font=("Segoe UI", 8, "bold"),
                 bg=self.PANEL_BG, fg=self.SUBTEXT).pack(pady=(8,6))

        # Camera selector
        cam_row = tk.Frame(frame, bg=self.PANEL_BG)
        cam_row.pack(fill="x", padx=12, pady=(0,6))
        tk.Label(cam_row, text="Camera:", font=("Segoe UI", 8),
                 bg=self.PANEL_BG, fg=self.TEXT).pack(side="left")
        for i in range(3):
            tk.Radiobutton(cam_row, text=str(i), variable=self.camera_index,
                           value=i, bg=self.PANEL_BG, fg=self.TEXT,
                           selectcolor=self.ACCENT,
                           activebackground=self.PANEL_BG,
                           font=("Segoe UI", 8)).pack(side="left", padx=3)

        # Start / Stop
        btn_row = tk.Frame(frame, bg=self.PANEL_BG)
        btn_row.pack(fill="x", padx=12, pady=(0,6))
        self.btn_start = tk.Button(
            btn_row, text="▶ Start", font=("Segoe UI", 9, "bold"),
            bg=self.GREEN, fg="#000000", activebackground="#00a865",
            relief="flat", padx=6, pady=4, cursor="hand2",
            command=self.start_detection)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0,3))

        self.btn_stop = tk.Button(
            btn_row, text="⏹ Stop", font=("Segoe UI", 9, "bold"),
            bg=self.RED, fg="white", activebackground="#b03040",
            relief="flat", padx=6, pady=4, cursor="hand2", state="disabled",
            command=self.stop_detection)
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=(3,0))

        # Clear alerts
        tk.Button(frame, text="Clear Alerts",
                  font=("Segoe UI", 8), bg=self.ACCENT, fg=self.TEXT,
                  activebackground="#1a4a80", relief="flat",
                  padx=6, pady=4, cursor="hand2",
                  command=self.clear_alerts).pack(fill="x", padx=8, pady=(0,8))

    def _build_log(self, parent):
        frame = tk.Frame(parent, bg=self.PANEL_BG)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="ALERT LOG", font=("Segoe UI", 8, "bold"),
                 bg=self.PANEL_BG, fg=self.SUBTEXT).pack(pady=(8,4))
        self.log_box = scrolledtext.ScrolledText(
            frame, bg=self.BG, fg=self.TEXT,
            font=("Consolas", 9), relief="flat",
            state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self.log_box.tag_config("alert", foreground=self.RED)
        self.log_box.tag_config("info",  foreground=self.SUBTEXT)

    # ── Controls ─────────────────────────────────────────────────────────────
    def start_detection(self):
        if self.running: return
        if not self._models_ready:
            self._log("Loading YOLOv8 models…", "info")
            self.root.update()
            self.yolo_model    = YOLO("yolov8l.pt")
            self.pose_model    = YOLO("yolov8l-pose.pt")
            self._models_ready = True
            self._log("Models loaded.", "info")

        cam = self.camera_index.get()
        # Use DirectShow backend on Windows for reliable camera access
        cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam)         # fallback: default backend
        if not cap.isOpened() and cam != 0:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._log("ERROR: Cannot open camera.", "alert"); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self._cap    = cap
        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_var.set(f"Running — camera {cam}")
        self._log(f"Detection started (camera {cam}).", "info")
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()

    def stop_detection(self):
        self.running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_var.set("Stopped.")
        self._log("Detection stopped.", "info")

    def clear_alerts(self):
        self.contaminated.clear()
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")
        self.lbl_persons.config(text="0")
        self.lbl_objects.config(text="0")
        self.lbl_contaminated.config(text="0")
        self._log("Alerts cleared.", "info")

    # ── Detection thread ─────────────────────────────────────────────────────
    def _detection_loop(self):
        cap = self._cap
        while self.running:
            ret, frame = cap.read()
            if not ret: break

            objs    = self.yolo_model.track(frame, persist=True,
                                            tracker="bytetrack.yaml",
                                            classes=OBJECTS, conf=0.25,
                                            verbose=False)[0]
            persons = self.yolo_model.track(frame, persist=True,
                                            tracker="bytetrack.yaml",
                                            classes=PERSONS, conf=0.5,
                                            verbose=False)[0]
            pose    = self.pose_model(frame, conf=0.3, verbose=False)[0]

            obj_boxes, person_boxes = [], []
            if objs.boxes is not None and len(objs.boxes):
                ids = (objs.boxes.id.cpu().numpy().astype(int)
                       if objs.boxes.id is not None else range(len(objs.boxes)))
                for box, cls, tid in zip(objs.boxes.xyxy.cpu().numpy(),
                                         objs.boxes.cls.cpu().numpy().astype(int), ids):
                    obj_boxes.append((tid, cls, box))

            if persons.boxes is not None and len(persons.boxes):
                ids = (persons.boxes.id.cpu().numpy().astype(int)
                       if persons.boxes.id is not None else range(len(persons.boxes)))
                for box, cls, tid in zip(persons.boxes.xyxy.cpu().numpy(),
                                         persons.boxes.cls.cpu().numpy().astype(int), ids):
                    person_boxes.append((tid, cls, box))

            palms = []
            if pose.keypoints is not None:
                for i in range(len(pose.keypoints)):
                    kp = pose.keypoints[i].data[0].cpu().numpy()
                    for p in get_palms(kp, frame.shape):
                        p['person'] = None
                        for pid, _, pbox in person_boxes:
                            if (pbox[0] <= p['center'][0] <= pbox[2] and
                                    pbox[1] <= p['center'][1] <= pbox[3]):
                                p['person'] = pid; break
                        palms.append(p)

            for palm in palms:
                for oid, ocls, obox in obj_boxes:
                    if iou(palm['box'], obox) > 0.10 and oid not in self.contaminated:
                        ts = time.strftime("%H:%M:%S")
                        self.contaminated[oid] = {'person': palm.get('person','?'),
                                                   'class': ocls, 'time': ts}
                        self.alert_queue.put(
                            f"[{ts}] Person {palm.get('person','?')} touched "
                            f"{CLASS_NAMES.get(ocls,'obj')} (ID:{oid})")

            # Draw overlays
            for pid, _, box in person_boxes:
                x1,y1,x2,y2 = box.astype(int)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,220,0), 2)
                cv2.putText(frame, f"Person {pid}",
                            (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,220,0), 2)

            for oid, ocls, box in obj_boxes:
                x1,y1,x2,y2 = box.astype(int)
                is_c  = oid in self.contaminated
                color = (0,0,255) if is_c else (0,220,80)
                label = f"{CLASS_NAMES.get(ocls,'obj')} {'CONTAMINATED' if is_c else 'CLEAN'}"
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label,
                            (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for palm in palms:
                cv2.circle(frame, palm['center'], 10, (0,255,200), -1)

            # HUD
            cv2.rectangle(frame, (0,0), (frame.shape[1], 36), (0,0,0), -1)
            cv2.putText(frame,
                        f"  Persons:{len(person_boxes)}  "
                        f"Objects:{len(obj_boxes)}  "
                        f"Contaminated:{len(self.contaminated)}",
                        (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

            if not self.frame_queue.full():
                self.frame_queue.put((frame.copy(),
                                      len(person_boxes),
                                      len(obj_boxes),
                                      len(self.contaminated)))
        cap.release()

    # ── tkinter polling loop ──────────────────────────────────────────────────
    def _poll(self):
        if not self.frame_queue.empty():
            frame, np_, no_, nc_ = self.frame_queue.get_nowait()
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((480, 360)))
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            self.lbl_persons.config(text=str(np_))
            self.lbl_objects.config(text=str(no_))
            self.lbl_contaminated.config(text=str(nc_))

        while not self.alert_queue.empty():
            self._log(self.alert_queue.get_nowait(), "alert")

        self.root.after(30, self._poll)

    def _log(self, msg: str, tag: str = "info"):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n", tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    ContactTracingApp(root)
    root.mainloop()

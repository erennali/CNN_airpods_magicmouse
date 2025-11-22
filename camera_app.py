"""
Advanced Multi-Model Object Detection System
Eren Ali Koca - 2212721021
BLG-407 Machine Learning Project
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import time
import threading
import queue
import os

class ObjectDetector:
    
    @staticmethod
    def detect_object_contours(image, min_area=2000):
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Daha agresif blur ve threshold
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            
            # Otsu's thresholding ile daha iyi segmentasyon
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Adaptive threshold ile de dene
            adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 15, 3)
            
            # İki threshold'u birleştir
            combined = cv2.bitwise_or(thresh, adaptive)
            
            # Daha büyük kernel ile morfolojik işlemler
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
            dilated = cv2.dilate(closed, kernel, iterations=2)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # En büyük contour'u al
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) < min_area:
                return None
            
            # Minimum area rectangle kullan (daha iyi fit)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Bounding box hesapla
            x, y, w, h = cv2.boundingRect(box)
            
            # Daha fazla padding ekle
            padding_x = int(w * 0.25)
            padding_y = int(h * 0.25)
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(image.shape[1] - x, w + 2 * padding_x)
            h = min(image.shape[0] - y, h + 2 * padding_y)
            
            return (x, y, w, h)
        
        except Exception as e:
            print(f"Contour detection error: {e}")
            return None
    
    @staticmethod
    def detect_object_center_focus(image, margin=0.10):
        try:
            h, w = image.shape[:2]
            x = int(w * margin)
            y = int(h * margin)
            box_w = int(w * (1 - 2 * margin))
            box_h = int(h * (1 - 2 * margin))
            return (x, y, box_w, box_h)
        except:
            return None
    
    @staticmethod
    def detect_object_smart(image):
        bbox = ObjectDetector.detect_object_contours(image)
        if bbox is None:
            bbox = ObjectDetector.detect_object_center_focus(image)
        return bbox


class MultiModelDetectionApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Model Object Detection - Eren Ali Koca")
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1400, int(screen_width * 0.85))
        window_height = min(900, int(screen_height * 0.85))
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1200, 700)
        self.root.resizable(True, True)
        
        self.root.configure(bg='#1a1a1a')
        
        self.models = {}
        self.models_info = {
            'Model 1 (VGG16)': {'path': 'model1_transfer_learning.h5', 'img_size': 224, 'color': '#00ff88'},
            'Model 2 (Basic CNN)': {'path': 'model2_basic_cnn.h5', 'img_size': 128, 'color': '#ff6b6b'},
            'Model 3 (Optimized CNN)': {'path': 'model3_improved_cnn.h5', 'img_size': 128, 'color': '#4dabf7'}
        }
        self.class_names = ['AirPods', 'Magic Mouse']
        self.current_image = None
        self.cap = None
        self.is_running = False
        self.camera_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.processing_lock = threading.Lock()
        self.prediction_times = {name: [] for name in self.models_info.keys()}
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        self.mode = tk.StringVar(value="single")
        self.selected_model = tk.StringVar(value="Model 1 (VGG16)")
        self.detection_method = tk.StringVar(value="Smart")
        self.show_bbox = tk.BooleanVar(value=True)
        
        self.setup_ui()
        self.load_models()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        main_container = tk.Frame(self.root, bg='#1a1a1a')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        header = tk.Frame(main_container, bg='#2d2d2d', height=80)
        header.pack(fill=tk.X, pady=(0, 15))
        header.pack_propagate(False)
        
        title_label = tk.Label(header, text="🎯 Multi-Model Object Detection System",
                              font=('Helvetica', 22, 'bold'), fg='#00ff88', bg='#2d2d2d')
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        self.status_label = tk.Label(header, text="⏳ Loading models...",
                                     font=('Helvetica', 12), fg='#ffa500', bg='#2d2d2d')
        self.status_label.pack(side=tk.RIGHT, padx=20)
        
        content_frame = tk.Frame(main_container, bg='#1a1a1a')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        video_panel = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.RIDGE, bd=2)
        video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        video_header = tk.Frame(video_panel, bg='#3d3d3d', height=50)
        video_header.pack(fill=tk.X)
        video_header.pack_propagate(False)
        
        tk.Label(video_header, text="📹 Live Detection", font=('Helvetica', 14, 'bold'),
                fg='#ffffff', bg='#3d3d3d').pack(side=tk.LEFT, padx=15, pady=10)
        
        self.fps_label = tk.Label(video_header, text="FPS: 0", font=('Helvetica', 11),
                                  fg='#00ff88', bg='#3d3d3d')
        self.fps_label.pack(side=tk.RIGHT, padx=15)
        
        self.video_frame = tk.Label(video_panel, bg='#000000')
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_panel = tk.Frame(content_frame, bg='#2d2d2d', width=360, relief=tk.RIDGE, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=0)
        right_panel.pack_propagate(False)
        
        # Create canvas with scrollbar for right panel
        self.canvas = tk.Canvas(right_panel, bg='#2d2d2d', highlightthickness=0)
        scrollbar = tk.Scrollbar(right_panel, orient="vertical", command=self.canvas.yview, 
                                bg='#3d3d3d', troughcolor='#2d2d2d', width=12)
        scrollable_frame = tk.Frame(self.canvas, bg='#2d2d2d')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=345)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        controls_frame = tk.Frame(scrollable_frame, bg='#2d2d2d')
        controls_frame.pack(fill=tk.X, padx=10, pady=8)
        
        tk.Label(controls_frame, text="⚙️ Controls", font=('Helvetica', 10, 'bold'),
                fg='#00ff88', bg='#2d2d2d').pack(anchor=tk.W, pady=(0, 5))
        
        # Buttons in a compact grid layout
        button_frame = tk.Frame(controls_frame, bg='#2d2d2d')
        button_frame.pack(fill=tk.X, pady=2)
        
        self.start_button = tk.Button(button_frame, text="▶ START", command=self.start_camera,
                                     font=('Helvetica', 8, 'bold'), fg='#000000', bg='#40c057',
                                     activebackground='#51cf66', activeforeground='#000000',
                                     relief=tk.FLAT, cursor='hand2',
                                     padx=5, pady=6, highlightthickness=0)
        self.start_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        
        self.stop_button = tk.Button(button_frame, text="⏹ STOP", command=self.stop_camera,
                                    font=('Helvetica', 8, 'bold'), fg='#000000', bg='#fa5252',
                                    activebackground='#ff6b6b', activeforeground='#000000',
                                    relief=tk.FLAT, cursor='hand2',
                                    padx=5, pady=6, state=tk.DISABLED, highlightthickness=0)
        self.stop_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))
        
        upload_frame = tk.Frame(controls_frame, bg='#2d2d2d')
        upload_frame.pack(fill=tk.X, pady=2)
        
        self.upload_button = tk.Button(upload_frame, text="📤 UPLOAD IMAGE", command=self.upload_image,
                                      font=('Helvetica', 8, 'bold'), fg='#000000', bg='#339af0',
                                      activebackground='#4dabf7', activeforeground='#000000',
                                      relief=tk.FLAT, cursor='hand2',
                                      padx=5, pady=6, highlightthickness=0)
        self.upload_button.pack(fill=tk.X)
        
        detection_frame = tk.Frame(scrollable_frame, bg='#2d2d2d')
        detection_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(detection_frame, text="🎯 Detection", font=('Helvetica', 10, 'bold'),
                fg='#4dabf7', bg='#2d2d2d').pack(anchor=tk.W, pady=(0, 4))
        
        method_frame = tk.Frame(detection_frame, bg='#3d3d3d', relief=tk.FLAT)
        method_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(method_frame, text="Method:", font=('Helvetica', 8),
                fg='#cccccc', bg='#3d3d3d').pack(anchor=tk.W, padx=6, pady=(4, 2))
        
        # Radio buttons in horizontal layout to save space
        radio_container = tk.Frame(method_frame, bg='#3d3d3d')
        radio_container.pack(fill=tk.X, padx=6, pady=2)
        
        methods = ["Smart", "Contour", "Center"]
        for method in methods:
            rb = tk.Radiobutton(radio_container, text=method, variable=self.detection_method,
                              value=method, font=('Helvetica', 8), fg='#ffffff', bg='#3d3d3d',
                              selectcolor='#1a1a1a', activebackground='#3d3d3d',
                              indicatoron=True)
            rb.pack(side=tk.LEFT, padx=3, pady=0)
        
        bbox_check = tk.Checkbutton(method_frame, text="Show Box", variable=self.show_bbox,
                                   font=('Helvetica', 8, 'bold'), fg='#00ff88', bg='#3d3d3d',
                                   selectcolor='#1a1a1a', activebackground='#3d3d3d')
        bbox_check.pack(anchor=tk.W, padx=6, pady=(3, 4))
        
        mode_frame = tk.Frame(scrollable_frame, bg='#2d2d2d')
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(mode_frame, text="🔄 Mode", font=('Helvetica', 10, 'bold'),
                fg='#ff6b6b', bg='#2d2d2d').pack(anchor=tk.W, pady=(0, 4))
        
        mode_buttons = tk.Frame(mode_frame, bg='#3d3d3d')
        mode_buttons.pack(fill=tk.X, pady=2)
        
        single_rb = tk.Radiobutton(mode_buttons, text="Single Model", variable=self.mode,
                                  value="single", command=self.on_mode_change,
                                  font=('Helvetica', 8, 'bold'), fg='#ffffff', bg='#3d3d3d',
                                  selectcolor='#1a1a1a', activebackground='#3d3d3d')
        single_rb.pack(anchor=tk.W, padx=6, pady=3)
        
        compare_rb = tk.Radiobutton(mode_buttons, text="Compare All", variable=self.mode,
                                   value="compare", command=self.on_mode_change,
                                   font=('Helvetica', 8, 'bold'), fg='#ffffff', bg='#3d3d3d',
                                   selectcolor='#1a1a1a', activebackground='#3d3d3d')
        compare_rb.pack(anchor=tk.W, padx=6, pady=3)
        
        model_frame = tk.Frame(scrollable_frame, bg='#2d2d2d')
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(model_frame, text="🤖 Model", font=('Helvetica', 10, 'bold'),
                fg='#ffa500', bg='#2d2d2d').pack(anchor=tk.W, pady=(0, 4))
        
        # Model dropdown with better visibility
        model_select_frame = tk.Frame(model_frame, bg='#3d3d3d')
        model_select_frame.pack(fill=tk.X, pady=2)
        
        self.model_selector = ttk.Combobox(model_select_frame, textvariable=self.selected_model,
                                          values=list(self.models_info.keys()),
                                          state='readonly', font=('Helvetica', 9))
        self.model_selector.pack(padx=6, pady=6, fill=tk.X)
        self.model_selector.bind('<<ComboboxSelected>>', lambda e: self.on_model_change())
        
        self.results_container = tk.Frame(scrollable_frame, bg='#2d2d2d')
        self.results_container.pack(fill=tk.X, padx=10, pady=5)
        
        self.setup_single_results()
        
        footer = tk.Frame(main_container, bg='#2d2d2d', height=40)
        footer.pack(fill=tk.X, pady=(15, 0))
        footer.pack_propagate(False)
        
        tk.Label(footer, text="© 2024 Eren Ali Koca | BLG-407 Machine Learning Project",
                font=('Helvetica', 9), fg='#888888', bg='#2d2d2d').pack(pady=10)
    
    def setup_single_results(self):
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        tk.Label(self.results_container, text="📊 Results", font=('Helvetica', 10, 'bold'),
                fg='#4dabf7', bg='#2d2d2d').pack(anchor=tk.W, pady=(0, 4))
        
        result_bg = tk.Frame(self.results_container, bg='#3d3d3d', relief=tk.FLAT, bd=1)
        result_bg.pack(fill=tk.X, pady=2)
        
        # Model info display - more compact
        self.model_info_label = tk.Label(result_bg, text=f"📌 {self.selected_model.get()}",
                                    font=('Helvetica', 8), fg='#ffa500', bg='#3d3d3d',
                                    anchor=tk.W)
        self.model_info_label.pack(fill=tk.X, padx=8, pady=(6, 8))
        
        # Separator
        tk.Frame(result_bg, bg='#555555', height=1).pack(fill=tk.X, padx=8, pady=2)
        
        self.prediction_label = tk.Label(result_bg, text="Waiting...",
                                        font=('Helvetica', 11, 'bold'), fg='#ffffff', bg='#3d3d3d',
                                        anchor=tk.W)
        self.prediction_label.pack(fill=tk.X, padx=8, pady=(6, 4))
        
        self.confidence_label = tk.Label(result_bg, text="Confidence: -",
                                        font=('Helvetica', 9), fg='#cccccc', bg='#3d3d3d',
                                        anchor=tk.W)
        self.confidence_label.pack(fill=tk.X, padx=8, pady=2)
        
        self.time_label = tk.Label(result_bg, text="Time: -",
                                  font=('Helvetica', 8), fg='#999999', bg='#3d3d3d',
                                  anchor=tk.W)
        self.time_label.pack(fill=tk.X, padx=8, pady=(2, 8))
    
    def setup_compare_results(self):
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        tk.Label(self.results_container, text="📊 Comparison", font=('Helvetica', 10, 'bold'),
                fg='#4dabf7', bg='#2d2d2d').pack(anchor=tk.W, pady=(0, 4))
        
        self.compare_labels = {}
        
        for model_name, info in self.models_info.items():
            model_frame = tk.Frame(self.results_container, bg='#3d3d3d', relief=tk.FLAT, bd=1)
            model_frame.pack(fill=tk.X, pady=2)
            
            # Compact header with colored indicator
            header_frame = tk.Frame(model_frame, bg='#3d3d3d')
            header_frame.pack(fill=tk.X, padx=6, pady=(4, 2))
            
            # Color indicator
            color_box = tk.Label(header_frame, text="■", font=('Helvetica', 12),
                               fg=info['color'], bg='#3d3d3d')
            color_box.pack(side=tk.LEFT, padx=(0, 4))
            
            header = tk.Label(header_frame, text=model_name.split('(')[0].strip(), 
                            font=('Helvetica', 9, 'bold'),
                            fg='#ffffff', bg='#3d3d3d', anchor=tk.W)
            header.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            pred_label = tk.Label(model_frame, text="Waiting...",
                                font=('Helvetica', 9), fg='#ffffff', bg='#3d3d3d', anchor=tk.W)
            pred_label.pack(fill=tk.X, padx=6, pady=1)
            
            conf_label = tk.Label(model_frame, text="Conf: -",
                                font=('Helvetica', 8), fg='#cccccc', bg='#3d3d3d', anchor=tk.W)
            conf_label.pack(fill=tk.X, padx=6, pady=1)
            
            time_label = tk.Label(model_frame, text="Time: -",
                                font=('Helvetica', 7), fg='#999999', bg='#3d3d3d', anchor=tk.W)
            time_label.pack(fill=tk.X, padx=6, pady=(1, 4))
            
            self.compare_labels[model_name] = {
                'prediction': pred_label,
                'confidence': conf_label,
                'time': time_label
            }
    
    def on_mode_change(self):
        if self.mode.get() == "single":
            self.setup_single_results()
            self.model_selector.config(state='readonly')
        else:
            self.setup_compare_results()
            self.model_selector.config(state='disabled')
    
    def on_model_change(self):
        # Update results display when model changes
        if self.mode.get() == "single" and hasattr(self, 'model_info_label'):
            self.model_info_label.config(text=f"📌 {self.selected_model.get()}")
            # Reset results when model changes
            self.prediction_label.config(text="Waiting...", fg='#ffffff')
            self.confidence_label.config(text="Confidence: -")
            self.time_label.config(text="Time: -")
    
    def load_models(self):
        loaded_count = 0
        failed_models = []
        
        for model_name, model_info in self.models_info.items():
            try:
                if os.path.exists(model_info['path']):
                    self.models[model_name] = load_model(model_info['path'], compile=False)
                    self.models[model_name].compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    loaded_count += 1
                    print(f"✅ Loaded: {model_name}")
                else:
                    failed_models.append(f"{model_name} (file not found)")
                    print(f"❌ Not found: {model_info['path']}")
            except Exception as e:
                failed_models.append(f"{model_name} ({str(e)[:30]}...)")
                print(f"❌ Failed to load {model_name}: {e}")
        
        if loaded_count == 3:
            self.status_label.config(
                text=f"✅ All models loaded successfully! ({loaded_count}/3)",
                fg='#00ff88'
            )
        elif loaded_count > 0:
            self.status_label.config(
                text=f"⚠️ Loaded {loaded_count}/3 models",
                fg='#ffa500'
            )
        else:
            self.status_label.config(
                text="❌ No models loaded",
                fg='#ff4444'
            )
            messagebox.showerror("Error", "Failed to load any models!")
    
    def find_object_bbox(self, image):
        method = self.detection_method.get()
        if method == "Contour":
            return ObjectDetector.detect_object_contours(image)
        elif method == "Center":
            return ObjectDetector.detect_object_center_focus(image)
        else:
            return ObjectDetector.detect_object_smart(image)
    
    def preprocess_image(self, image_input, target_size):
        try:
            if isinstance(image_input, np.ndarray):
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(image_input)
            else:
                pil_img = image_input
            
            pil_img = pil_img.resize(target_size, Image.LANCZOS)
            img_array = keras_image.img_to_array(pil_img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict_with_model(self, model_name, image_input):
        if model_name not in self.models:
            return None, 0, 0
        
        try:
            with self.processing_lock:
                model = self.models[model_name]
                img_size = self.models_info[model_name]['img_size']
                
                start_time = time.time()
                preprocessed = self.preprocess_image(image_input, (img_size, img_size))
                
                if preprocessed is None:
                    return None, 0, 0
                
                predictions = model.predict(preprocessed, verbose=0)
                inference_time = (time.time() - start_time) * 1000
                
                predicted_class_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class_idx] * 100
                predicted_class = self.class_names[predicted_class_idx]
                
                self.prediction_times[model_name].append(inference_time)
                if len(self.prediction_times[model_name]) > 100:
                    self.prediction_times[model_name].pop(0)
                
                return predicted_class, confidence, inference_time
        except Exception as e:
            print(f"Prediction error for {model_name}: {e}")
            return None, 0, 0
    
    def draw_bounding_box(self, image, bbox, label, confidence, color):
        if bbox is None or not self.show_bbox.get():
            return image
        
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        img_h, img_w, _ = image.shape
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            return image
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        
        corner_size = min(w, h) // 8
        corner_thickness = 4
        
        cv2.line(image, (x, y), (x + corner_size, y), color, corner_thickness)
        cv2.line(image, (x, y), (x, y + corner_size), color, corner_thickness)
        
        cv2.line(image, (x + w, y), (x + w - corner_size, y), color, corner_thickness)
        cv2.line(image, (x + w, y), (x + w, y + corner_size), color, corner_thickness)
        
        cv2.line(image, (x, y + h), (x + corner_size, y + h), color, corner_thickness)
        cv2.line(image, (x, y + h), (x, y + h - corner_size), color, corner_thickness)
        
        cv2.line(image, (x + w, y + h), (x + w - corner_size, y + h), color, corner_thickness)
        cv2.line(image, (x + w, y + h), (x + w, y + h - corner_size), color, corner_thickness)
        
        text = f"{label}: {confidence:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x
        text_y = y - 12 if y - 12 > text_size[1] else y + text_size[1] + 12
        
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 6),
                     (text_x + text_size[0] + 6, text_y + 6), color, -1)
        
        cv2.putText(image, text, (text_x + 3, text_y), font, font_scale,
                   (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        return image
    
    def update_single_result(self, prediction, confidence, inference_time):
        if prediction:
            self.prediction_label.config(text=f"✓ {prediction}",
                                        fg='#00ff88' if confidence > 70 else '#ffa500')
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
            self.time_label.config(text=f"Time: {inference_time:.0f}ms")
        else:
            self.prediction_label.config(text="⚠ Error", fg='#ff4444')
            self.confidence_label.config(text="Confidence: -")
            self.time_label.config(text="Time: -")
    
    def update_compare_results(self, results):
        for model_name, (pred, conf, time_ms) in results.items():
            if model_name in self.compare_labels:
                labels = self.compare_labels[model_name]
                if pred:
                    labels['prediction'].config(text=f"✓ {pred}")
                    labels['confidence'].config(text=f"Conf: {conf:.0f}%")
                    labels['time'].config(text=f"Time: {time_ms:.0f}ms")
                else:
                    labels['prediction'].config(text="⚠ Error")
                    labels['confidence'].config(text="Conf: -")
                    labels['time'].config(text="Time: -")
    
    def update_fps(self):
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            fps = self.fps_counter / elapsed
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def camera_thread_func(self):
        while self.is_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        if not self.frame_queue.full():
                            try:
                                self.frame_queue.put(frame, block=False)
                            except queue.Full:
                                pass
                    else:
                        print("Failed to read frame")
                        time.sleep(0.1)
                else:
                    break
            except Exception as e:
                print(f"Camera thread error: {e}")
                break
        print("Camera thread stopped")
    
    def process_frames(self):
        if not self.is_running:
            return
        
        try:
            frame = self.frame_queue.get_nowait()
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox = self.find_object_bbox(rgb_frame)
            
            if self.mode.get() == "single":
                model_name = self.selected_model.get()
                prediction, confidence, inference_time = self.predict_with_model(model_name, rgb_frame)
                
                if prediction:
                    color_hex = self.models_info[model_name]['color']
                    color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
                    display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    display_frame = self.draw_bounding_box(display_frame, bbox, prediction, confidence, color_bgr)
                    self.display_image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                    self.update_single_result(prediction, confidence, inference_time)
                else:
                    self.display_image(rgb_frame)
            else:
                results = {}
                for model_name in self.models_info.keys():
                    pred, conf, time_ms = self.predict_with_model(model_name, rgb_frame)
                    results[model_name] = (pred, conf, time_ms)
                
                display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                for model_name, (pred, conf, _) in results.items():
                    if pred:
                        color_hex = self.models_info[model_name]['color']
                        color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
                        display_frame = self.draw_bounding_box(display_frame, bbox, pred, conf, color_bgr)
                        break
                
                self.display_image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                self.update_compare_results(results)
            
            self.update_fps()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Frame processing error: {e}")
        
        if self.is_running:
            self.root.after(10, self.process_frames)
    
    def start_camera(self):
        if not self.models:
            messagebox.showerror("Error", "No models loaded!")
            return
        
        try:
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open camera!")
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.upload_button.config(state=tk.DISABLED)
            self.status_label.config(
                text="🎥 Camera active - Real-time detection...",
                fg='#00ff88'
            )
            
            self.camera_thread = threading.Thread(target=self.camera_thread_func, daemon=True)
            self.camera_thread.start()
            
            self.fps_counter = 0
            self.fps_start_time = time.time()
            self.process_frames()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")
            self.cleanup_camera()
    
    def cleanup_camera(self):
        try:
            if self.cap:
                self.cap.release()
            self.cap = None
        except Exception as e:
            print(f"Camera cleanup error: {e}")
        
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass
    
    def stop_camera(self):
        self.is_running = False
        
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        
        self.cleanup_camera()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.NORMAL)
        self.status_label.config(
            text="✅ Models loaded and ready",
            fg='#00ff88'
        )
        self.fps_label.config(text="FPS: 0")
        
        self.video_frame.config(image='', bg='#000000')
        print("Camera stopped")
    
    def upload_image(self):
        if not self.models:
            messagebox.showerror("Error", "No models loaded!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            image = Image.open(file_path)
            image_array = np.array(image)
            
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            bbox = self.find_object_bbox(image_array)
            
            if self.mode.get() == "single":
                model_name = self.selected_model.get()
                prediction, confidence, inference_time = self.predict_with_model(model_name, image_array)
                
                if prediction:
                    color_hex = self.models_info[model_name]['color']
                    color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
                    display_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    display_image = self.draw_bounding_box(display_image, bbox, prediction, confidence, color_bgr)
                    self.display_image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
                    self.update_single_result(prediction, confidence, inference_time)
                else:
                    self.display_image(image_array)
                    messagebox.showerror("Error", "Failed to process image")
            else:
                results = {}
                for model_name in self.models_info.keys():
                    pred, conf, time_ms = self.predict_with_model(model_name, image_array)
                    results[model_name] = (pred, conf, time_ms)
                
                display_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                for model_name, (pred, conf, _) in results.items():
                    if pred:
                        color_hex = self.models_info[model_name]['color']
                        color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
                        display_image = self.draw_bounding_box(display_image, bbox, pred, conf, color_bgr)
                        break
                
                self.display_image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
                self.update_compare_results(results)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")
            print(f"Upload error: {e}")
    
    def display_image(self, image):
        try:
            frame_width = self.video_frame.winfo_width()
            frame_height = self.video_frame.winfo_height()
            
            if frame_width <= 1 or frame_height <= 1:
                frame_width = 800
                frame_height = 600
            
            img_h, img_w = image.shape[:2]
            scale = min(frame_width / img_w, frame_height / img_h) * 0.95
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                pil_image = Image.fromarray(resized)
                photo = ImageTk.PhotoImage(image=pil_image)
                
                self.video_frame.config(image=photo)
                self.video_frame.image = photo
        except Exception as e:
            print(f"Display error: {e}")
    
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = MultiModelDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

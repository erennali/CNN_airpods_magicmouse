"""
Advanced Multi-Model Object Detection System
Eren Ali Koca - 2212721021
BLG-407 Machine Learning Project
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import threading
import queue
import time
import os
import sys

class ObjectDetector:

    @staticmethod
    def detect_object_contours(image, min_area=3000):
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

                blurred = cv2.GaussianBlur(gray, (9, 9), 0)
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                dilated = cv2.dilate(closed, kernel, iterations=1)

                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    return None

                    largest_contour = max(contours, key=cv2.contourArea)

                    if cv2.contourArea(largest_contour) < min_area:
                        return None

                        x, y, w, h = cv2.boundingRect(largest_contour)

                        padding_x = int(w * 0.15)
                        padding_y = int(h * 0.15)

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

                            except Exception as e:
                                print(f"Center focus detection error: {e}")
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
                                                self.root.title("🎯 Advanced Object Detection - CNN Model Comparison")

                                                # Get screen dimensions
                                                screen_width = root.winfo_screenwidth()
                                                screen_height = root.winfo_screenheight()

                                                # Set window size (90% of screen or minimum 1400x850)
                                                window_width = max(1400, int(screen_width * 0.9))
                                                window_height = max(850, int(screen_height * 0.9))

                                                # Center window on screen
                                                x = (screen_width - window_width) // 2
                                                y = (screen_height - window_height) // 2

                                                self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
                                                self.root.minsize(1200, 700)  # Minimum size
                                                self.root.resizable(True, True)  # Allow resizing
                                                self.root.configure(bg='#1a1a2e')

                                                # Model configurations
                                                self.models_info = {
                                                'Model 1 - VGG16': {
                                                'path': 'model1_transfer_learning.h5',
                                                'img_size': 224,
                                                'description': 'Transfer Learning (VGG16)',
                                                'color': (0, 255, 136)  # Parlak yeşil - RGB format
                                                },
                                                'Model 2 - Basic CNN': {
                                                'path': 'model2_basic_cnn.h5',
                                                'img_size': 128,
                                                'description': 'Basic CNN from Scratch',
                                                'color': (255, 100, 255)  # Parlak pembe/mor - RGB format
                                                },
                                                'Model 3 - Optimized': {
                                                'path': 'model3_improved_cnn.h5',
                                                'img_size': 128,
                                                'description': 'Optimized CNN + Augmentation',
                                                'color': (0, 255, 255)  # Parlak cyan - RGB format
                                                }
                                                }

                                                # State variables
                                                self.models = {}
                                                self.class_names = ['AirPods', 'Magic Mouse']
                                                self.cap = None
                                                self.is_running = False
                                                self.frame_queue = queue.Queue(maxsize=3)
                                                self.current_mode = 'single'
                                                self.selected_model = 'Model 1 - VGG16'
                                                self.show_bbox = tk.BooleanVar(value=True)
                                                self.detection_method = tk.StringVar(value='smart')

                                                # Threading
                                                self.camera_thread = None
                                                self.processing_lock = threading.Lock()

                                                # Performance tracking
                                                self.prediction_times = {name: [] for name in self.models_info.keys()}
                                                self.fps_counter = 0
                                                self.fps_start_time = time.time()
                                                self.current_fps = 0

                                                # Setup UI and load models
                                                self.setup_ui()
                                                self.load_models()

                                                def setup_ui(self):
                                                    # Main container
                                                    main_container = tk.Frame(self.root, bg='#1a1a2e')
                                                    main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                                                    # Header
                                                    self._create_header(main_container)

                                                    # Content area
                                                    content_frame = tk.Frame(main_container, bg='#1a1a2e')
                                                    content_frame.pack(fill=tk.BOTH, expand=True)

                                                    # Left panel - Video/Image display
                                                    self._create_video_panel(content_frame)

                                                    # Right panel - Controls and Results
                                                    self._create_control_panel(content_frame)

                                                    # Footer
                                                    self._create_footer(main_container)

                                                    def _create_header(self, parent):
                                                        header_frame = tk.Frame(parent, bg='#16213e', relief=tk.RAISED, bd=2)
                                                        header_frame.pack(fill=tk.X, pady=(0, 10))

                                                        header_title = tk.Label(
                                                        header_frame,
                                                        text="🎯 ADVANCED OBJECT DETECTION SYSTEM",
                                                        font=('Helvetica', 20, 'bold'),
                                                        bg='#16213e',
                                                        fg='#ffffff'
                                                        )
                                                        header_title.pack(pady=10)

                                                        subtitle = tk.Label(
                                                        header_frame,
                                                        text="Model 1 (VGG16) • Model 2 (Basic CNN) • Model 3 (Optimized) | Real-time Bounding Box Detection",
                                                        font=('Helvetica', 9),
                                                        bg='#16213e',
                                                        fg='#aaaaaa'
                                                        )
                                                        subtitle.pack(pady=(0, 10))

                                                        def _create_video_panel(self, parent):
                                                            left_panel = tk.Frame(parent, bg='#16213e', relief=tk.RAISED, bd=2)
                                                            left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

                                                            video_header = tk.Label(
                                                            left_panel,
                                                            text="📹 Real-time Detection",
                                                            font=('Helvetica', 12, 'bold'),
                                                            bg='#16213e',
                                                            fg='#ffffff'
                                                            )
                                                            video_header.pack(pady=8)

                                                            # FPS display
                                                            self.fps_label = tk.Label(
                                                            left_panel,
                                                            text="FPS: 0",
                                                            font=('Helvetica', 9),
                                                            bg='#16213e',
                                                            fg='#ffd700'
                                                            )
                                                            self.fps_label.pack()

                                                            # Video frame with dynamic sizing
                                                            self.video_frame = tk.Label(left_panel, bg='#000000')
                                                            self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                                                            def _create_control_panel(self, parent):
                                                                right_panel = tk.Frame(parent, bg='#16213e', width=380, relief=tk.RAISED, bd=2)
                                                                right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
                                                                right_panel.pack_propagate(False)

                                                                # Detection settings (compact)
                                                                self._create_detection_settings(right_panel)

                                                                # Mode selection (compact)
                                                                self._create_mode_selection(right_panel)

                                                                # Model selection (for single mode)
                                                                self._create_model_selection(right_panel)

                                                                # Control buttons - MOVED UP SO ALWAYS VISIBLE!
                                                                self._create_control_buttons(right_panel)

                                                                # Status
                                                                self.status_label = tk.Label(
                                                                right_panel,
                                                                text="⏳ Models loading...",
                                                                font=('Helvetica', 8),
                                                                bg='#16213e',
                                                                fg='#ffd700',
                                                                wraplength=350
                                                                )
                                                                self.status_label.pack(pady=4)

                                                                # Results area (takes remaining space)
                                                                self.results_frame = tk.Frame(right_panel, bg='#16213e')
                                                                self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

                                                                self.create_single_results_ui()

                                                                def _create_detection_settings(self, parent):
                                                                    settings_frame = tk.LabelFrame(
                                                                    parent,
                                                                    text="🔍 Settings",
                                                                    font=('Helvetica', 9, 'bold'),
                                                                    bg='#16213e',
                                                                    fg='#ffffff',
                                                                    relief=tk.GROOVE,
                                                                    bd=2
                                                                    )
                                                                    settings_frame.pack(fill=tk.X, padx=10, pady=4)

                                                                    # Bounding box toggle
                                                                    bbox_check = tk.Checkbutton(
                                                                    settings_frame,
                                                                    text="Show Bounding Box",
                                                                    variable=self.show_bbox,
                                                                    font=('Helvetica', 8),
                                                                    bg='#16213e',
                                                                    fg='#ffffff',
                                                                    selectcolor='#0f3460',
                                                                    activebackground='#16213e',
                                                                    activeforeground='#ffffff'
                                                                    )
                                                                    bbox_check.pack(anchor='w', padx=6, pady=2)

                                                                    # Detection method - compact
                                                                    methods = [
                                                                    ('Smart', 'smart'),
                                                                    ('Contour', 'contour'),
                                                                    ('Center', 'center')
                                                                    ]

                                                                    method_frame = tk.Frame(settings_frame, bg='#16213e')
                                                                    method_frame.pack(fill=tk.X, padx=6, pady=2)

                                                                    for text, value in methods:
                                                                        rb = tk.Radiobutton(
                                                                        method_frame,
                                                                        text=text,
                                                                        variable=self.detection_method,
                                                                        value=value,
                                                                        font=('Helvetica', 7),
                                                                        bg='#16213e',
                                                                        fg='#ffffff',
                                                                        selectcolor='#0f3460',
                                                                        activebackground='#16213e',
                                                                        activeforeground='#ffffff'
                                                                        )
                                                                        rb.pack(side=tk.LEFT, padx=4)

                                                                        def _create_mode_selection(self, parent):
                                                                            mode_frame = tk.LabelFrame(
                                                                            parent,
                                                                            text="⚙️ Mode",
                                                                            font=('Helvetica', 9, 'bold'),
                                                                            bg='#16213e',
                                                                            fg='#ffffff',
                                                                            relief=tk.GROOVE,
                                                                            bd=2
                                                                            )
                                                                            mode_frame.pack(fill=tk.X, padx=10, pady=4)

                                                                            self.mode_var = tk.StringVar(value='single')

                                                                            single_mode_radio = tk.Radiobutton(
                                                                            mode_frame,
                                                                            text="🔍 Single",
                                                                            variable=self.mode_var,
                                                                            value='single',
                                                                            command=self.change_mode,
                                                                            font=('Helvetica', 8),
                                                                            bg='#16213e',
                                                                            fg='#ffffff',
                                                                            selectcolor='#0f3460',
                                                                            activebackground='#16213e',
                                                                            activeforeground='#ffffff'
                                                                            )
                                                                            single_mode_radio.pack(anchor='w', padx=6, pady=2)

                                                                            compare_mode_radio = tk.Radiobutton(
                                                                            mode_frame,
                                                                            text="⚖️ Compare All",
                                                                            variable=self.mode_var,
                                                                            value='compare',
                                                                            command=self.change_mode,
                                                                            font=('Helvetica', 8),
                                                                            bg='#16213e',
                                                                            fg='#ffffff',
                                                                            selectcolor='#0f3460',
                                                                            activebackground='#16213e',
                                                                            activeforeground='#ffffff'
                                                                            )
                                                                            compare_mode_radio.pack(anchor='w', padx=6, pady=2)

                                                                            def _create_model_selection(self, parent):
                                                                                self.model_select_frame = tk.LabelFrame(
                                                                                parent,
                                                                                text="🤖 Model",
                                                                                font=('Helvetica', 9, 'bold'),
                                                                                bg='#16213e',
                                                                                fg='#ffffff',
                                                                                relief=tk.GROOVE,
                                                                                bd=2
                                                                                )
                                                                                self.model_select_frame.pack(fill=tk.X, padx=10, pady=4)

                                                                                self.model_combo = ttk.Combobox(
                                                                                self.model_select_frame,
                                                                                values=list(self.models_info.keys()),
                                                                                state='readonly',
                                                                                font=('Helvetica', 8),
                                                                                width=28
                                                                                )
                                                                                self.model_combo.set(self.selected_model)
                                                                                self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
                                                                                self.model_combo.pack(padx=6, pady=4)

                                                                                def _create_control_buttons(self, parent):
                                                                                    button_container = tk.LabelFrame(
                                                                                    parent,
                                                                                    text="🎮 Controls",
                                                                                    font=('Helvetica', 10, 'bold'),
                                                                                    bg='#16213e',
                                                                                    fg='#ffffff',
                                                                                    relief=tk.GROOVE,
                                                                                    bd=2
                                                                                    )
                                                                                    button_container.pack(fill=tk.X, padx=10, pady=6)

                                                                                    button_frame = tk.Frame(button_container, bg='#16213e')
                                                                                    button_frame.pack(pady=6, padx=6)

                                                                                    self.start_button = tk.Button(
                                                                                    button_frame,
                                                                                    text="▶ Start Camera",
                                                                                    command=self.start_camera,
                                                                                    font=('Helvetica', 10, 'bold'),
                                                                                    bg='#00ff88',  # Çok daha parlak yeşil!
                                                                                    fg='#000000',  # Siyah yazı daha okunaklı
                                                                                    activebackground='#33ffaa',
                                                                                    activeforeground='#000000',
                                                                                    relief=tk.RAISED,
                                                                                    width=20,
                                                                                    height=2,
                                                                                    cursor='hand2',
                                                                                    bd=3
                                                                                    )
                                                                                    self.start_button.pack(pady=4)

                                                                                    self.stop_button = tk.Button(
                                                                                    button_frame,
                                                                                    text="⏸ Stop Camera",
                                                                                    command=self.stop_camera,
                                                                                    font=('Helvetica', 10, 'bold'),
                                                                                    bg='#ff6666',  # Çok daha parlak kırmızı!
                                                                                    fg='#000000',  # Siyah yazı
                                                                                    activebackground='#ff8888',
                                                                                    activeforeground='#000000',
                                                                                    relief=tk.RAISED,
                                                                                    width=20,
                                                                                    height=2,
                                                                                    cursor='hand2',
                                                                                    state=tk.DISABLED,
                                                                                    bd=3
                                                                                    )
                                                                                    self.stop_button.pack(pady=4)

                                                                                    separator = ttk.Separator(button_frame, orient='horizontal')
                                                                                    separator.pack(fill='x', pady=5)

                                                                                    self.upload_button = tk.Button(
                                                                                    button_frame,
                                                                                    text="📁 Upload Image",
                                                                                    command=self.upload_image,
                                                                                    font=('Helvetica', 10, 'bold'),
                                                                                    bg='#aa66ff',  # Çok daha parlak mor!
                                                                                    fg='#000000',  # Siyah yazı
                                                                                    activebackground='#cc88ff',
                                                                                    activeforeground='#000000',
                                                                                    relief=tk.RAISED,
                                                                                    width=20,
                                                                                    height=2,
                                                                                    cursor='hand2',
                                                                                    bd=3
                                                                                    )
                                                                                    self.upload_button.pack(pady=4)

                                                                                    def _create_footer(self, parent):
                                                                                        footer = tk.Label(
                                                                                        parent,
                                                                                        text="Eren Ali Koca - 2212721021 | BLG-407 Machine Learning Project | Advanced Object Detection with CNN",
                                                                                        font=('Helvetica', 8),
                                                                                        bg='#1a1a2e',
                                                                                        fg='#888888'
                                                                                        )
                                                                                        footer.pack(pady=(6, 0))

                                                                                        def create_single_results_ui(self):
                                                                                            for widget in self.results_frame.winfo_children():
                                                                                                widget.destroy()

                                                                                                result_container = tk.Frame(self.results_frame, bg='#0f3460', relief=tk.RAISED, bd=2)
                                                                                                result_container.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

                                                                                                result_title = tk.Label(
                                                                                                result_container,
                                                                                                text="📊 PREDICTION RESULT",
                                                                                                font=('Helvetica', 11, 'bold'),
                                                                                                bg='#0f3460',
                                                                                                fg='#ffffff'
                                                                                                )
                                                                                                result_title.pack(pady=8)

                                                                                                self.single_prediction_label = tk.Label(
                                                                                                result_container,
                                                                                                text="-",
                                                                                                font=('Helvetica', 22, 'bold'),
                                                                                                bg='#0f3460',
                                                                                                fg='#00ff88'
                                                                                                )
                                                                                                self.single_prediction_label.pack(pady=6)

                                                                                                conf_label = tk.Label(
                                                                                                result_container,
                                                                                                text="CONFIDENCE:",
                                                                                                font=('Helvetica', 9, 'bold'),
                                                                                                bg='#0f3460',
                                                                                                fg='#cccccc'
                                                                                                )
                                                                                                conf_label.pack(pady=(6, 3))

                                                                                                self.single_confidence_label = tk.Label(
                                                                                                result_container,
                                                                                                text="-",
                                                                                                font=('Helvetica', 18, 'bold'),
                                                                                                bg='#0f3460',
                                                                                                fg='#00aaff'
                                                                                                )
                                                                                                self.single_confidence_label.pack(pady=3)

                                                                                                self.single_progress = ttk.Progressbar(
                                                                                                result_container,
                                                                                                length=300,
                                                                                                mode='determinate'
                                                                                                )
                                                                                                self.single_progress.pack(pady=10)

                                                                                                time_label = tk.Label(
                                                                                                result_container,
                                                                                                text="INFERENCE TIME:",
                                                                                                font=('Helvetica', 9, 'bold'),
                                                                                                bg='#0f3460',
                                                                                                fg='#cccccc'
                                                                                                )
                                                                                                time_label.pack(pady=(6, 3))

                                                                                                self.single_time_label = tk.Label(
                                                                                                result_container,
                                                                                                text="-",
                                                                                                font=('Helvetica', 12),
                                                                                                bg='#0f3460',
                                                                                                fg='#ffd700'
                                                                                                )
                                                                                                self.single_time_label.pack(pady=3)

                                                                                                # Detection info
                                                                                                detection_label = tk.Label(
                                                                                                result_container,
                                                                                                text="DETECTION:",
                                                                                                font=('Helvetica', 9, 'bold'),
                                                                                                bg='#0f3460',
                                                                                                fg='#cccccc'
                                                                                                )
                                                                                                detection_label.pack(pady=(6, 3))

                                                                                                self.detection_info_label = tk.Label(
                                                                                                result_container,
                                                                                                text="-",
                                                                                                font=('Helvetica', 8),
                                                                                                bg='#0f3460',
                                                                                                fg='#aaaaaa',
                                                                                                wraplength=340
                                                                                                )
                                                                                                self.detection_info_label.pack(pady=3)

                                                                                                def create_compare_results_ui(self):
                                                                                                    for widget in self.results_frame.winfo_children():
                                                                                                        widget.destroy()

                                                                                                        compare_title = tk.Label(
                                                                                                        self.results_frame,
                                                                                                        text="⚖️ MODEL COMPARISON",
                                                                                                        font=('Helvetica', 10, 'bold'),
                                                                                                        bg='#16213e',
                                                                                                        fg='#ffffff'
                                                                                                        )
                                                                                                        compare_title.pack(pady=6)

                                                                                                        # Scrollable frame
                                                                                                        canvas = tk.Canvas(self.results_frame, bg='#16213e', highlightthickness=0)
                                                                                                        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=canvas.yview)
                                                                                                        scrollable_frame = tk.Frame(canvas, bg='#16213e')

                                                                                                        scrollable_frame.bind(
                                                                                                        "<Configure>",
                                                                                                        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
                                                                                                        )

                                                                                                        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                                                                                                        canvas.configure(yscrollcommand=scrollbar.set)

                                                                                                        self.compare_widgets = {}

                                                                                                        for model_name, model_info in self.models_info.items():
                                                                                                            model_frame = tk.Frame(scrollable_frame, bg='#0f3460', relief=tk.RAISED, bd=2)
                                                                                                            model_frame.pack(fill=tk.X, padx=3, pady=3)

                                                                                                            # Convert RGB to hex for Tkinter
                                                                                                            rgb = model_info['color']
                                                                                                            hex_color = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

                                                                                                            title = tk.Label(
                                                                                                            model_frame,
                                                                                                            text=f"🤖 {model_name}",
                                                                                                            font=('Helvetica', 9, 'bold'),
                                                                                                            bg='#0f3460',
                                                                                                            fg=hex_color
                                                                                                            )
                                                                                                            title.pack(pady=4)

                                                                                                            desc = tk.Label(
                                                                                                            model_frame,
                                                                                                            text=model_info['description'],
                                                                                                            font=('Helvetica', 7),
                                                                                                            bg='#0f3460',
                                                                                                            fg='#aaaaaa'
                                                                                                            )
                                                                                                            desc.pack()

                                                                                                            pred_label = tk.Label(
                                                                                                            model_frame,
                                                                                                            text="-",
                                                                                                            font=('Helvetica', 13, 'bold'),
                                                                                                            bg='#0f3460',
                                                                                                            fg='#ffffff'
                                                                                                            )
                                                                                                            pred_label.pack(pady=3)

                                                                                                            conf_label = tk.Label(
                                                                                                            model_frame,
                                                                                                            text="-",
                                                                                                            font=('Helvetica', 11),
                                                                                                            bg='#0f3460',
                                                                                                            fg='#00aaff'
                                                                                                            )
                                                                                                            conf_label.pack()

                                                                                                            time_label = tk.Label(
                                                                                                            model_frame,
                                                                                                            text="-",
                                                                                                            font=('Helvetica', 8),
                                                                                                            bg='#0f3460',
                                                                                                            fg='#ffd700'
                                                                                                            )
                                                                                                            time_label.pack(pady=3)

                                                                                                            self.compare_widgets[model_name] = {
                                                                                                            'prediction': pred_label,
                                                                                                            'confidence': conf_label,
                                                                                                            'time': time_label,
                                                                                                            'frame': model_frame
                                                                                                            }

                                                                                                            canvas.pack(side="left", fill="both", expand=True)
                                                                                                            scrollbar.pack(side="right", fill="y")

                                                                                                            def change_mode(self):
                                                                                                                self.current_mode = self.mode_var.get()

                                                                                                                if self.current_mode == 'single':
                                                                                                                    self.model_select_frame.pack(fill=tk.X, padx=10, pady=4, after=self.mode_var.master)
                                                                                                                    self.create_single_results_ui()
                                                                                                                else:
                                                                                                                    self.model_select_frame.pack_forget()
                                                                                                                    self.create_compare_results_ui()

                                                                                                                    self.status_label.config(
                                                                                                                    text=f"✅ Mode: {'Single' if self.current_mode == 'single' else 'Compare'}",
                                                                                                                    fg='#00ff88'
                                                                                                                    )

                                                                                                                    def on_model_change(self, event):
                                                                                                                        self.selected_model = self.model_combo.get()
                                                                                                                        self.status_label.config(
                                                                                                                        text=f"✅ Selected model: {self.selected_model}",
                                                                                                                        fg='#00ff88'
                                                                                                                        )

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

                                                                                                                                        # Update status
                                                                                                                                        if loaded_count == 3:
                                                                                                                                            self.status_label.config(
                                                                                                                                            text=f"✅ All models loaded successfully! ({loaded_count}/3)",
                                                                                                                                            fg='#00ff88'
                                                                                                                                            )
                                                                                                                                        elif loaded_count > 0:
                                                                                                                                            self.status_label.config(
                                                                                                                                            text=f"⚠️ {loaded_count}/3 models loaded\nFailed: {', '.join(failed_models)}",
                                                                                                                                            fg='#ffa500'
                                                                                                                                            )
                                                                                                                                        else:
                                                                                                                                            self.status_label.config(
                                                                                                                                            text=f"❌ No models loaded!\nFailed: {', '.join(failed_models)}",
                                                                                                                                            fg='#ff4444'
                                                                                                                                            )
                                                                                                                                            messagebox.showerror(
                                                                                                                                            "Model Loading Error",
                                                                                                                                            "No models could be loaded. Please check model files exist."
                                                                                                                                            )

                                                                                                                                            def detect_object_bbox(self, image):
                                                                                                                                                method = self.detection_method.get()

                                                                                                                                                if method == 'smart':
                                                                                                                                                    return ObjectDetector.detect_object_smart(image)
                                                                                                                                                elif method == 'contour':
                                                                                                                                                    return ObjectDetector.detect_object_contours(image)
                                                                                                                                                elif method == 'center':
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
                                                                                                                                                                                        inference_time = (time.time() - start_time) * 1000  # ms

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

                                                                                                                                                                                                    try:
                                                                                                                                                                                                        x, y, w, h = bbox

                                                                                                                                                                                                        # Draw main rectangle - THICKER!
                                                                                                                                                                                                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)

                                                                                                                                                                                                        # Draw corner markers (more professional look) - LONGER and THICKER!
                                                                                                                                                                                                        corner_length = int(min(w, h) * 0.15)  # 15% of smallest dimension
                                                                                                                                                                                                        corner_thickness = 6

                                                                                                                                                                                                        # Top-left
                                                                                                                                                                                                        cv2.line(image, (x, y), (x + corner_length, y), color, corner_thickness)
                                                                                                                                                                                                        cv2.line(image, (x, y), (x, y + corner_length), color, corner_thickness)
                                                                                                                                                                                                        # Top-right
                                                                                                                                                                                                        cv2.line(image, (x + w, y), (x + w - corner_length, y), color, corner_thickness)
                                                                                                                                                                                                        cv2.line(image, (x + w, y), (x + w, y + corner_length), color, corner_thickness)
                                                                                                                                                                                                        # Bottom-left
                                                                                                                                                                                                        cv2.line(image, (x, y + h), (x + corner_length, y + h), color, corner_thickness)
                                                                                                                                                                                                        cv2.line(image, (x, y + h), (x, y + h - corner_length), color, corner_thickness)
                                                                                                                                                                                                        # Bottom-right
                                                                                                                                                                                                        cv2.line(image, (x + w, y + h), (x + w - corner_length, y + h), color, corner_thickness)
                                                                                                                                                                                                        cv2.line(image, (x + w, y + h), (x + w, y + h - corner_length), color, corner_thickness)

                                                                                                                                                                                                        # Draw label background - BIGGER!
                                                                                                                                                                                                        label_text = f"{label}: {confidence:.1f}%"
                                                                                                                                                                                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                                                                                                                                                                                        font_scale = 0.9  # Bigger text
                                                                                                                                                                                                        thickness = 2
                                                                                                                                                                                                        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

                                                                                                                                                                                                        # Background rectangle with padding
                                                                                                                                                                                                        padding = 8
                                                                                                                                                                                                        cv2.rectangle(image,
                                                                                                                                                                                                        (x - 2, y - text_h - padding * 2),
                                                                                                                                                                                                        (x + text_w + padding * 2, y),
                                                                                                                                                                                                        color, -1)

                                                                                                                                                                                                        # Add border to label
                                                                                                                                                                                                        cv2.rectangle(image,
                                                                                                                                                                                                        (x - 2, y - text_h - padding * 2),
                                                                                                                                                                                                        (x + text_w + padding * 2, y),
                                                                                                                                                                                                        (0, 0, 0), 2)

                                                                                                                                                                                                        # Text - WHITE and BOLD
                                                                                                                                                                                                        cv2.putText(image, label_text, (x + padding, y - padding),
                                                                                                                                                                                                        font, font_scale, (255, 255, 255), thickness + 1)

                                                                                                                                                                                                        return image

                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                        print(f"Drawing error: {e}")
                                                                                                                                                                                                        return image

                                                                                                                                                                                                        def update_single_display(self, predicted_class, confidence, inference_time, bbox_info=""):
                                                                                                                                                                                                            try:
                                                                                                                                                                                                                self.single_prediction_label.config(text=predicted_class.upper())
                                                                                                                                                                                                                self.single_confidence_label.config(text=f"{confidence:.1f}%")
                                                                                                                                                                                                                self.single_progress['value'] = min(confidence, 100)
                                                                                                                                                                                                                self.single_time_label.config(text=f"{inference_time:.1f} ms")

                                                                                                                                                                                                                if bbox_info:
                                                                                                                                                                                                                    self.detection_info_label.config(text=bbox_info)

                                                                                                                                                                                                                    # Color coding based on confidence
                                                                                                                                                                                                                    if confidence > 80:
                                                                                                                                                                                                                        color = '#00ff88'
                                                                                                                                                                                                                    elif confidence > 60:
                                                                                                                                                                                                                        color = '#ffa500'
                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                        color = '#ff4444'

                                                                                                                                                                                                                        self.single_prediction_label.config(fg=color)
                                                                                                                                                                                                                        self.single_confidence_label.config(fg=color)

                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                        print(f"Display update error: {e}")

                                                                                                                                                                                                                        def update_compare_display(self, results):
                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                best_conf = 0
                                                                                                                                                                                                                                best_model = None

                                                                                                                                                                                                                                for model_name, result in results.items():
                                                                                                                                                                                                                                    pred, conf, inf_time = result
                                                                                                                                                                                                                                    if conf > best_conf:
                                                                                                                                                                                                                                        best_conf = conf
                                                                                                                                                                                                                                        best_model = model_name

                                                                                                                                                                                                                                        for model_name, result in results.items():
                                                                                                                                                                                                                                            if model_name in self.compare_widgets:
                                                                                                                                                                                                                                                widgets = self.compare_widgets[model_name]
                                                                                                                                                                                                                                                pred, conf, inf_time = result

                                                                                                                                                                                                                                                if pred:
                                                                                                                                                                                                                                                    widgets['prediction'].config(text=pred.upper())
                                                                                                                                                                                                                                                    widgets['confidence'].config(text=f"{conf:.1f}%")
                                                                                                                                                                                                                                                    widgets['time'].config(text=f"⏱ {inf_time:.1f} ms")

                                                                                                                                                                                                                                                    # Highlight best prediction
                                                                                                                                                                                                                                                    if model_name == best_model:
                                                                                                                                                                                                                                                        widgets['frame'].config(bg='#1a5490', bd=3)
                                                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                        widgets['frame'].config(bg='#0f3460', bd=2)
                                                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                        widgets['prediction'].config(text="ERROR")
                                                                                                                                                                                                                                                        widgets['confidence'].config(text="-")
                                                                                                                                                                                                                                                        widgets['time'].config(text="-")

                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                        print(f"Compare display update error: {e}")

                                                                                                                                                                                                                                                        def update_fps(self):
                                                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                                                self.fps_counter += 1
                                                                                                                                                                                                                                                                elapsed = time.time() - self.fps_start_time

                                                                                                                                                                                                                                                                if elapsed >= 1.0:
                                                                                                                                                                                                                                                                    self.current_fps = self.fps_counter / elapsed
                                                                                                                                                                                                                                                                    self.fps_label.config(text=f"FPS: {self.current_fps:.1f}")
                                                                                                                                                                                                                                                                    self.fps_counter = 0
                                                                                                                                                                                                                                                                    self.fps_start_time = time.time()

                                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                                    print(f"FPS update error: {e}")

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

                                                                                                                                                                                                                                                                                                time.sleep(0.01)  # Small delay to prevent CPU overuse

                                                                                                                                                                                                                                                                                                def process_frames(self):
                                                                                                                                                                                                                                                                                                    if not self.is_running:
                                                                                                                                                                                                                                                                                                        return

                                                                                                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                                                                                                            frame = self.frame_queue.get_nowait()
                                                                                                                                                                                                                                                                                                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                                                                                                                                                                                                                                                                                                            # Detect object bbox
                                                                                                                                                                                                                                                                                                            bbox = self.detect_object_bbox(frame_rgb)
                                                                                                                                                                                                                                                                                                            bbox_info = ""

                                                                                                                                                                                                                                                                                                            if bbox:
                                                                                                                                                                                                                                                                                                                x, y, w, h = bbox
                                                                                                                                                                                                                                                                                                                bbox_info = f"Object: {w}x{h}px at ({x},{y})"

                                                                                                                                                                                                                                                                                                                if self.current_mode == 'single':
                                                                                                                                                                                                                                                                                                                    # Single model prediction
                                                                                                                                                                                                                                                                                                                    pred, conf, inf_time = self.predict_with_model(self.selected_model, frame_rgb)

                                                                                                                                                                                                                                                                                                                    if pred:
                                                                                                                                                                                                                                                                                                                        self.update_single_display(pred, conf, inf_time, bbox_info)

                                                                                                                                                                                                                                                                                                                        # Draw bounding box
                                                                                                                                                                                                                                                                                                                        color = self.models_info[self.selected_model]['color']
                                                                                                                                                                                                                                                                                                                        frame_rgb = self.draw_bounding_box(frame_rgb, bbox, pred, conf, color)

                                                                                                                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                                                                                        # Compare mode - predict with all models
                                                                                                                                                                                                                                                                                                                        results = {}
                                                                                                                                                                                                                                                                                                                        for model_name in self.models_info.keys():
                                                                                                                                                                                                                                                                                                                            results[model_name] = self.predict_with_model(model_name, frame_rgb)

                                                                                                                                                                                                                                                                                                                            self.update_compare_display(results)

                                                                                                                                                                                                                                                                                                                            # Draw bounding box for best prediction
                                                                                                                                                                                                                                                                                                                            best_model = max(results.keys(),
                                                                                                                                                                                                                                                                                                                            key=lambda k: results[k][1] if results[k][0] else 0)
                                                                                                                                                                                                                                                                                                                            pred, conf, inf_time = results[best_model]

                                                                                                                                                                                                                                                                                                                            if pred and bbox:
                                                                                                                                                                                                                                                                                                                                color = self.models_info[best_model]['color']
                                                                                                                                                                                                                                                                                                                                frame_rgb = self.draw_bounding_box(frame_rgb, bbox,
                                                                                                                                                                                                                                                                                                                                f"{best_model.split('-')[0]}: {pred}",
                                                                                                                                                                                                                                                                                                                                conf, color)

                                                                                                                                                                                                                                                                                                                                # Draw all model predictions as text overlay
                                                                                                                                                                                                                                                                                                                                y_offset = 35
                                                                                                                                                                                                                                                                                                                                for model_name, (pred, conf, inf_time) in results.items():
                                                                                                                                                                                                                                                                                                                                    if pred:
                                                                                                                                                                                                                                                                                                                                        color = self.models_info[model_name]['color']
                                                                                                                                                                                                                                                                                                                                        text = f"{model_name.split('-')[0]}: {pred} ({conf:.0f}%)"
                                                                                                                                                                                                                                                                                                                                        cv2.putText(frame_rgb, text, (10, y_offset),
                                                                                                                                                                                                                                                                                                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                                                                                                                                                                                                                                                                                                                        y_offset += 30

                                                                                                                                                                                                                                                                                                                                        # Update FPS
                                                                                                                                                                                                                                                                                                                                        self.update_fps()

                                                                                                                                                                                                                                                                                                                                        # Display frame with dynamic sizing
                                                                                                                                                                                                                                                                                                                                        self.display_image(frame_rgb)

                                                                                                                                                                                                                                                                                                                                    except queue.Empty:
                                                                                                                                                                                                                                                                                                                                        pass
                                                                                                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                                                                                                        print(f"Frame processing error: {e}")
                                                                                                                                                                                                                                                                                                                                        import traceback
                                                                                                                                                                                                                                                                                                                                        traceback.print_exc()

                                                                                                                                                                                                                                                                                                                                        # Schedule next frame
                                                                                                                                                                                                                                                                                                                                        if self.is_running:
                                                                                                                                                                                                                                                                                                                                            self.root.after(10, self.process_frames)

                                                                                                                                                                                                                                                                                                                                            def start_camera(self):
                                                                                                                                                                                                                                                                                                                                                if not self.models:
                                                                                                                                                                                                                                                                                                                                                    messagebox.showerror("Error", "No models loaded!")
                                                                                                                                                                                                                                                                                                                                                    return

                                                                                                                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                                                                                                                        # Try to open camera
                                                                                                                                                                                                                                                                                                                                                        self.cap = cv2.VideoCapture(0)

                                                                                                                                                                                                                                                                                                                                                        if not self.cap.isOpened():
                                                                                                                                                                                                                                                                                                                                                            messagebox.showerror("Error", "Cannot open camera!")
                                                                                                                                                                                                                                                                                                                                                            return

                                                                                                                                                                                                                                                                                                                                                            # Set camera properties for better performance
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

                                                                                                                                                                                                                                                                                                                                                            # Start camera thread
                                                                                                                                                                                                                                                                                                                                                            self.camera_thread = threading.Thread(target=self.camera_thread_func, daemon=True)
                                                                                                                                                                                                                                                                                                                                                            self.camera_thread.start()

                                                                                                                                                                                                                                                                                                                                                            # Start frame processing
                                                                                                                                                                                                                                                                                                                                                            self.fps_counter = 0
                                                                                                                                                                                                                                                                                                                                                            self.fps_start_time = time.time()
                                                                                                                                                                                                                                                                                                                                                            self.process_frames()

                                                                                                                                                                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                                                                                                                                                                            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
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

                                                                                                                                                                                                                                                                                                                                                                                    # Wait for thread to finish
                                                                                                                                                                                                                                                                                                                                                                                    if self.camera_thread and self.camera_thread.is_alive():
                                                                                                                                                                                                                                                                                                                                                                                        self.camera_thread.join(timeout=1.0)

                                                                                                                                                                                                                                                                                                                                                                                        # Cleanup
                                                                                                                                                                                                                                                                                                                                                                                        self.cleanup_camera()

                                                                                                                                                                                                                                                                                                                                                                                        # Update UI
                                                                                                                                                                                                                                                                                                                                                                                        self.start_button.config(state=tk.NORMAL)
                                                                                                                                                                                                                                                                                                                                                                                        self.stop_button.config(state=tk.DISABLED)
                                                                                                                                                                                                                                                                                                                                                                                        self.upload_button.config(state=tk.NORMAL)
                                                                                                                                                                                                                                                                                                                                                                                        self.status_label.config(
                                                                                                                                                                                                                                                                                                                                                                                        text="⏸ Camera stopped",
                                                                                                                                                                                                                                                                                                                                                                                        fg='#ffa500'
                                                                                                                                                                                                                                                                                                                                                                                        )
                                                                                                                                                                                                                                                                                                                                                                                        self.fps_label.config(text="FPS: 0")

                                                                                                                                                                                                                                                                                                                                                                                        # Reset displays
                                                                                                                                                                                                                                                                                                                                                                                        if self.current_mode == 'single':
                                                                                                                                                                                                                                                                                                                                                                                            self.single_prediction_label.config(text="-")
                                                                                                                                                                                                                                                                                                                                                                                            self.single_confidence_label.config(text="-")
                                                                                                                                                                                                                                                                                                                                                                                            self.single_progress['value'] = 0
                                                                                                                                                                                                                                                                                                                                                                                            self.single_time_label.config(text="-")
                                                                                                                                                                                                                                                                                                                                                                                            self.detection_info_label.config(text="-")
                                                                                                                                                                                                                                                                                                                                                                                        else:
                                                                                                                                                                                                                                                                                                                                                                                            for widgets in self.compare_widgets.values():
                                                                                                                                                                                                                                                                                                                                                                                                widgets['prediction'].config(text="-")
                                                                                                                                                                                                                                                                                                                                                                                                widgets['confidence'].config(text="-")
                                                                                                                                                                                                                                                                                                                                                                                                widgets['time'].config(text="-")

                                                                                                                                                                                                                                                                                                                                                                                                def upload_image(self):
                                                                                                                                                                                                                                                                                                                                                                                                    if not self.models:
                                                                                                                                                                                                                                                                                                                                                                                                        messagebox.showerror("Error", "No models loaded!")
                                                                                                                                                                                                                                                                                                                                                                                                        return

                                                                                                                                                                                                                                                                                                                                                                                                        file_path = filedialog.askopenfilename(
                                                                                                                                                                                                                                                                                                                                                                                                        title="Select Image",
                                                                                                                                                                                                                                                                                                                                                                                                        filetypes=[
                                                                                                                                                                                                                                                                                                                                                                                                        ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                                                                                                                                                                                                                                                                                                                                                                                                        ("All Files", "*.*")
                                                                                                                                                                                                                                                                                                                                                                                                        ]
                                                                                                                                                                                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                                                                                                                                                                        if not file_path:
                                                                                                                                                                                                                                                                                                                                                                                                            return

                                                                                                                                                                                                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                                                                                                                                                                                                self.stop_camera()

                                                                                                                                                                                                                                                                                                                                                                                                                # Load image
                                                                                                                                                                                                                                                                                                                                                                                                                img_pil = Image.open(file_path).convert('RGB')
                                                                                                                                                                                                                                                                                                                                                                                                                img_array = np.array(img_pil)

                                                                                                                                                                                                                                                                                                                                                                                                                # Detect object bbox
                                                                                                                                                                                                                                                                                                                                                                                                                bbox = self.detect_object_bbox(img_array)
                                                                                                                                                                                                                                                                                                                                                                                                                bbox_info = ""

                                                                                                                                                                                                                                                                                                                                                                                                                if bbox:
                                                                                                                                                                                                                                                                                                                                                                                                                    x, y, w, h = bbox
                                                                                                                                                                                                                                                                                                                                                                                                                    bbox_info = f"Object detected: {w}x{h}px at ({x},{y})"
                                                                                                                                                                                                                                                                                                                                                                                                                else:
                                                                                                                                                                                                                                                                                                                                                                                                                    bbox_info = "No clear object boundary detected"

                                                                                                                                                                                                                                                                                                                                                                                                                    if self.current_mode == 'single':
                                                                                                                                                                                                                                                                                                                                                                                                                        # Single model prediction
                                                                                                                                                                                                                                                                                                                                                                                                                        pred, conf, inf_time = self.predict_with_model(self.selected_model, img_pil)

                                                                                                                                                                                                                                                                                                                                                                                                                        if pred:
                                                                                                                                                                                                                                                                                                                                                                                                                            self.update_single_display(pred, conf, inf_time, bbox_info)

                                                                                                                                                                                                                                                                                                                                                                                                                            # Draw bounding box
                                                                                                                                                                                                                                                                                                                                                                                                                            img_display = img_array.copy()
                                                                                                                                                                                                                                                                                                                                                                                                                            color = self.models_info[self.selected_model]['color']
                                                                                                                                                                                                                                                                                                                                                                                                                            img_display = self.draw_bounding_box(img_display, bbox, pred, conf, color)

                                                                                                                                                                                                                                                                                                                                                                                                                            # Display
                                                                                                                                                                                                                                                                                                                                                                                                                            self.display_image(img_display)

                                                                                                                                                                                                                                                                                                                                                                                                                        else:
                                                                                                                                                                                                                                                                                                                                                                                                                            # Compare mode
                                                                                                                                                                                                                                                                                                                                                                                                                            results = {}
                                                                                                                                                                                                                                                                                                                                                                                                                            for model_name in self.models_info.keys():
                                                                                                                                                                                                                                                                                                                                                                                                                                results[model_name] = self.predict_with_model(model_name, img_pil)

                                                                                                                                                                                                                                                                                                                                                                                                                                self.update_compare_display(results)

                                                                                                                                                                                                                                                                                                                                                                                                                                # Draw bounding box for best model
                                                                                                                                                                                                                                                                                                                                                                                                                                img_display = img_array.copy()

                                                                                                                                                                                                                                                                                                                                                                                                                                best_model = max(results.keys(),
                                                                                                                                                                                                                                                                                                                                                                                                                                key=lambda k: results[k][1] if results[k][0] else 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                pred, conf, inf_time = results[best_model]

                                                                                                                                                                                                                                                                                                                                                                                                                                if pred and bbox:
                                                                                                                                                                                                                                                                                                                                                                                                                                    color = self.models_info[best_model]['color']
                                                                                                                                                                                                                                                                                                                                                                                                                                    img_display = self.draw_bounding_box(img_display, bbox,
                                                                                                                                                                                                                                                                                                                                                                                                                                    f"{best_model.split('-')[0]}: {pred}",
                                                                                                                                                                                                                                                                                                                                                                                                                                    conf, color)

                                                                                                                                                                                                                                                                                                                                                                                                                                    # Draw all predictions as text
                                                                                                                                                                                                                                                                                                                                                                                                                                    y_offset = 35
                                                                                                                                                                                                                                                                                                                                                                                                                                    for model_name, (pred, conf, inf_time) in results.items():
                                                                                                                                                                                                                                                                                                                                                                                                                                        if pred:
                                                                                                                                                                                                                                                                                                                                                                                                                                            color = self.models_info[model_name]['color']
                                                                                                                                                                                                                                                                                                                                                                                                                                            text = f"{model_name.split('-')[0]}: {pred} ({conf:.0f}%)"
                                                                                                                                                                                                                                                                                                                                                                                                                                            cv2.putText(img_display, text, (10, y_offset),
                                                                                                                                                                                                                                                                                                                                                                                                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                            y_offset += 35

                                                                                                                                                                                                                                                                                                                                                                                                                                            # Display
                                                                                                                                                                                                                                                                                                                                                                                                                                            self.display_image(img_display)

                                                                                                                                                                                                                                                                                                                                                                                                                                            self.status_label.config(
                                                                                                                                                                                                                                                                                                                                                                                                                                            text=f"📁 Image loaded and tested\n{bbox_info}",
                                                                                                                                                                                                                                                                                                                                                                                                                                            fg='#8844ff'
                                                                                                                                                                                                                                                                                                                                                                                                                                            )

                                                                                                                                                                                                                                                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                                            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
                                                                                                                                                                                                                                                                                                                                                                                                                                            import traceback
                                                                                                                                                                                                                                                                                                                                                                                                                                            traceback.print_exc()

                                                                                                                                                                                                                                                                                                                                                                                                                                            def display_image(self, img_array):
                                                                                                                                                                                                                                                                                                                                                                                                                                                try:
                                                                                                                                                                                                                                                                                                                                                                                                                                                    img = Image.fromarray(img_array)

                                                                                                                                                                                                                                                                                                                                                                                                                                                    # Get video frame size
                                                                                                                                                                                                                                                                                                                                                                                                                                                    frame_width = self.video_frame.winfo_width()
                                                                                                                                                                                                                                                                                                                                                                                                                                                    frame_height = self.video_frame.winfo_height()

                                                                                                                                                                                                                                                                                                                                                                                                                                                    # Use default size if frame not yet rendered
                                                                                                                                                                                                                                                                                                                                                                                                                                                    if frame_width <= 1 or frame_height <= 1:
                                                                                                                                                                                                                                                                                                                                                                                                                                                        frame_width = 800
                                                                                                                                                                                                                                                                                                                                                                                                                                                        frame_height = 600

                                                                                                                                                                                                                                                                                                                                                                                                                                                        # Calculate aspect ratio preserving size
                                                                                                                                                                                                                                                                                                                                                                                                                                                        img_ratio = img.width / img.height
                                                                                                                                                                                                                                                                                                                                                                                                                                                        frame_ratio = frame_width / frame_height

                                                                                                                                                                                                                                                                                                                                                                                                                                                        if img_ratio > frame_ratio:
                                                                                                                                                                                                                                                                                                                                                                                                                                                            # Image is wider
                                                                                                                                                                                                                                                                                                                                                                                                                                                            new_width = frame_width - 20
                                                                                                                                                                                                                                                                                                                                                                                                                                                            new_height = int(new_width / img_ratio)
                                                                                                                                                                                                                                                                                                                                                                                                                                                        else:
                                                                                                                                                                                                                                                                                                                                                                                                                                                            # Image is taller
                                                                                                                                                                                                                                                                                                                                                                                                                                                            new_height = frame_height - 20
                                                                                                                                                                                                                                                                                                                                                                                                                                                            new_width = int(new_height * img_ratio)

                                                                                                                                                                                                                                                                                                                                                                                                                                                            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                                                                                                                                                                                                                                                                                                                                                                                                                                            imgtk = ImageTk.PhotoImage(image=img)
                                                                                                                                                                                                                                                                                                                                                                                                                                                            self.video_frame.imgtk = imgtk
                                                                                                                                                                                                                                                                                                                                                                                                                                                            self.video_frame.configure(image=imgtk)
                                                                                                                                                                                                                                                                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                                                            print(f"Display error: {e}")

                                                                                                                                                                                                                                                                                                                                                                                                                                                            def on_closing(self):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                print("Closing application...")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                self.is_running = False

                                                                                                                                                                                                                                                                                                                                                                                                                                                                # Wait for threads
                                                                                                                                                                                                                                                                                                                                                                                                                                                                if self.camera_thread and self.camera_thread.is_alive():
                                                                                                                                                                                                                                                                                                                                                                                                                                                                    self.camera_thread.join(timeout=2.0)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # Cleanup camera
                                                                                                                                                                                                                                                                                                                                                                                                                                                                    self.cleanup_camera()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        cv2.destroyAllWindows()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                    except:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        pass

                                                                                                                                                                                                                                                                                                                                                                                                                                                                        self.root.destroy()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        print("Application closed successfully")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                        def main():
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            try:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                root = tk.Tk()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                app = MultiModelDetectionApp(root)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                root.protocol("WM_DELETE_WINDOW", app.on_closing)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                root.mainloop()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                print(f"Critical error: {e}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                import traceback
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                traceback.print_exc()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                messagebox.showerror("Critical Error", f"Application failed: {str(e)}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                sys.exit(1)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    main()

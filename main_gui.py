import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import cv2
import numpy as np
import joblib
import os

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

MODEL_PATH = "model_knn.pkl"
SCALER_PATH = "scaler.pkl"

SIDEBAR_COLOR = "#2c3e50"
GRADIENT_START = "#EDF2F7"
GRADIENT_END = "#FFFFFF"
TEXT_WHITE = "#FFFFFF"
TEXT_DARK = "#333333"


def create_vertical_gradient(width, height, start_color, end_color):
    """Membuat gambar PIL dengan gradasi vertikal."""
    base = Image.new('RGB', (width, height), start_color)
    top = Image.new('RGB', (width, height), end_color)
    mask = Image.new('L', (width, height))
    mask_data = []
    for y in range(height):
        mask_data.extend([int(255 * (y / height))] * width)
    mask.putdata(mask_data)
    gradient = Image.composite(top, base, mask)
    return gradient


class RoseClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistem Klasifikasi Mawar | Skripsi 2026")
        self.geometry("1100x720")
        self.set_icon_setup()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # FIX 1: Initialize image references to prevent garbage collection
        self.bg_ctk_image = None
        self.photo = None
        self.current_image_path = None

        self.check_model_files()
        self.setup_sidebar()
        self.setup_main_dashboard()

    def set_icon_setup(self):
        try:
            self.iconbitmap("icon.ico")
        except Exception:
            pass

    def check_model_files(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            messagebox.showwarning(
                "Error",
                "Model/Scaler tidak ditemukan! Jalankan train_model.py dulu."
            )
            self.model = None
            self.scaler = None
        else:
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memuat model: {e}")
                self.model = None
                self.scaler = None

    def setup_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=270, corner_radius=0, fg_color=SIDEBAR_COLOR)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="KLASIFIKASI\nBUNGA MAWAR MERAH\nKUNING, PUTIH",
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
            text_color=TEXT_WHITE
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(50, 30))

        self.btn_upload = ctk.CTkButton(
            self.sidebar_frame, text="📂 PILIH CITRA", height=50,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#34495e", hover_color="#415b76",
            command=self.upload_image
        )
        self.btn_upload.grid(row=1, column=0, padx=25, pady=15, sticky="ew")

        self.btn_process = ctk.CTkButton(
            self.sidebar_frame, text="⚡ MULAI ANALISIS", height=50,
            fg_color="#27ae60", hover_color="#2ecc71",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.predict_image
        )
        self.btn_process.grid(row=2, column=0, padx=25, pady=15, sticky="ew")

        self.lbl_footer = ctk.CTkLabel(
            self.sidebar_frame,
            text="Metode: K-Nearest Neighbor\nSkripsi TA 2026",
            font=ctk.CTkFont(size=11), text_color="#bdc3c7"
        )
        self.lbl_footer.grid(row=5, column=0, padx=20, pady=20)

    def setup_main_dashboard(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew")

        self.bg_gradient_label = ctk.CTkLabel(self.main_frame, text="")
        self.bg_gradient_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.main_frame.bind("<Configure>", self.resize_background)

        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=30, pady=30)

        self.lbl_header = ctk.CTkLabel(
            self.content_frame,
            text="Dashboard Analisis & Hasil",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color=SIDEBAR_COLOR
        )
        self.lbl_header.pack(anchor="w", pady=(0, 25))

        self.image_frame = ctk.CTkFrame(
            self.content_frame, fg_color="#FAFAFA",
            corner_radius=20, border_width=1, border_color="#E0E0E0"
        )
        self.image_frame.pack(fill="both", expand=True, pady=(0, 20))

        self.lbl_image = ctk.CTkLabel(
            self.image_frame,
            text="[Silakan Pilih Citra Mawar di Sidebar]",
            font=ctk.CTkFont(size=14), text_color="gray"
        )
        self.lbl_image.place(relx=0.5, rely=0.5, anchor="center")

        # FIX 4: Use grid layout for result_container to avoid overlapping labels
        self.result_container = ctk.CTkFrame(
            self.content_frame, height=220, corner_radius=20,
            fg_color="white", border_width=0
        )
        self.result_container.pack(fill="x", pady=(10, 0))
        self.result_container.pack_propagate(False)
        self.result_container.grid_columnconfigure(0, weight=2)
        self.result_container.grid_columnconfigure(1, weight=2)
        self.result_container.grid_columnconfigure(2, weight=2)
        self.result_container.grid_rowconfigure(0, weight=1)

        # Column 0 — Prediction result
        left_frame = ctk.CTkFrame(self.result_container, fg_color="transparent")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=15)

        self.lbl_result = ctk.CTkLabel(
            left_frame, text="...",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color=TEXT_DARK
        )
        self.lbl_result.pack(anchor="w", pady=(20, 5))

        self.lbl_confidence = ctk.CTkLabel(
            left_frame, text="Confidence Level: -%",
            font=ctk.CTkFont(size=14), text_color="gray"
        )
        self.lbl_confidence.pack(anchor="w")

        # Column 1 — Probability distribution
        mid_frame = ctk.CTkFrame(
            self.result_container, fg_color="#F4F6F7",
            corner_radius=15
        )
        mid_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=15)

        ctk.CTkLabel(
            mid_frame,
            text="Distribusi Probabilitas (KNN Vote):",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=(15, 5))

        self.lbl_stats_detail = ctk.CTkLabel(
            mid_frame, text="-\n-\n-",
            font=ctk.CTkFont(family="Consolas", size=13),
            justify="left", text_color=SIDEBAR_COLOR
        )
        self.lbl_stats_detail.pack(pady=(0, 10))

        # Column 2 — Feature extraction info
        right_frame = ctk.CTkFrame(self.result_container, fg_color="transparent")
        right_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 20), pady=15)

        ctk.CTkLabel(
            right_frame,
            text="Ekstraksi Fitur Digital:",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="gray"
        ).pack(anchor="w", pady=(20, 8))

        self.lbl_rgb = ctk.CTkLabel(
            right_frame, text="RGB Mean: -",
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.lbl_rgb.pack(anchor="w", pady=4)

        self.lbl_shape = ctk.CTkLabel(
            right_frame, text="Hu Moment (Shape): -",
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.lbl_shape.pack(anchor="w", pady=4)

    def resize_background(self, event):
        """Fungsi untuk membuat ulang gradasi saat jendela di-resize."""
        width = event.width
        height = event.height
        if width < 10 or height < 10:
            return

        pil_image = create_vertical_gradient(width, height, GRADIENT_START, GRADIENT_END)

        # FIX 1: Store image reference on self to prevent garbage collection
        self.bg_ctk_image = ctk.CTkImage(
            light_image=pil_image, dark_image=pil_image, size=(width, height)
        )
        self.bg_gradient_label.configure(image=self.bg_ctk_image)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        self.current_image_path = file_path

        img = Image.open(file_path)

        # FIX 3: Call update_idletasks() before reading widget dimensions
        self.update_idletasks()
        container_width = self.image_frame.winfo_width() - 20
        container_height = self.image_frame.winfo_height() - 20
        if container_width < 100:
            container_width = 400
        if container_height < 100:
            container_height = 300

        img_ratio = img.width / img.height
        container_ratio = container_width / container_height

        if img_ratio > container_ratio:
            new_width = container_width
            new_height = int(new_width / img_ratio)
        else:
            new_height = container_height
            new_width = int(new_height * img_ratio)

        img = img.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)

        # FIX 1: Store photo reference on self to prevent garbage collection
        self.photo = ctk.CTkImage(
            light_image=img, dark_image=img, size=(new_width, new_height)
        )
        self.lbl_image.configure(image=self.photo, text="")

        # Reset result labels
        self.lbl_result.configure(text="...", text_color=TEXT_DARK)
        self.lbl_confidence.configure(text="Confidence Level: -%")
        self.lbl_stats_detail.configure(text="-\n-\n-")
        self.lbl_rgb.configure(text="RGB Mean: -")
        self.lbl_shape.configure(text="Hu Moment (Shape): -")

    def extract_features_gui(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None

        img_resized = cv2.resize(img, (300, 300))

        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        mask_daun = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        mask_bukan_daun = cv2.bitwise_not(mask_daun)

        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_mask = cv2.bitwise_and(mask_otsu, mask_otsu, mask=mask_bukan_daun)

        # FIX 6: Guard against empty mask producing zero/NaN features
        if cv2.countNonZero(final_mask) == 0:
            messagebox.showwarning(
                "Peringatan",
                "Segmentasi gambar gagal (mask kosong).\n"
                "Coba gunakan gambar dengan latar belakang lebih kontras."
            )
            return None

        mean_val = cv2.mean(img_resized, mask=final_mask)[:3]
        hu = cv2.HuMoments(cv2.moments(final_mask)).flatten()
        hu_log = [
            -1 * np.sign(h) * np.log10(np.abs(h)) if h != 0 else 0
            for h in hu
        ]

        return [mean_val[2], mean_val[1], mean_val[0]] + hu_log

    def predict_image(self):
        # FIX 5: Guard against missing model, scaler, or image
        if not self.model or not self.scaler:
            messagebox.showwarning("Warning", "Model atau Scaler belum dimuat.")
            return
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Harap pilih gambar mawar terlebih dahulu di sidebar.")
            return

        try:
            features = self.extract_features_gui(self.current_image_path)
            if features is None:
                return

            print("\n" + "=" * 40)
            print(f"ANALISIS FILE: {os.path.basename(self.current_image_path)}")
            print("-" * 20)
            print(f"R: {features[0]:.2f} | G: {features[1]:.2f} | B: {features[2]:.2f}")
            print("Hu-Moments (1-7):")
            for i, h in enumerate(features[3:], 1):
                print(f"  Hu{i}: {h:.6f}")
            print("=" * 40 + "\n")

            features_scaled = self.scaler.transform(np.array([features]))

            prediction = self.model.predict(features_scaled)[0]

            # FIX 2: Handle models that may not support predict_proba
            try:
                proba = self.model.predict_proba(features_scaled)[0]
                classes = self.model.classes_
                max_prob = np.max(proba) * 100
            except AttributeError:
                messagebox.showerror(
                    "Error",
                    "Model tidak mendukung predict_proba.\n"
                    "Pastikan KNeighborsClassifier diinisialisasi dengan benar."
                )
                return

            label_clean = prediction.replace("_", " ").upper()

            color_map = {
                "MAWAR MERAH": "#e74c3c",
                "MAWAR PUTIH": "#95a5a6",
                "MAWAR KUNING": "#f1c40f"
            }
            res_color = color_map.get(label_clean, TEXT_DARK)

            self.lbl_result.configure(text=label_clean, text_color=res_color)
            self.lbl_confidence.configure(text=f"Confidence Level: {max_prob:.1f}%")

            detail_text = ""
            for i, class_name in enumerate(classes):
                clean_name = class_name.replace("_", " ").title()
                percent = proba[i] * 100
                prefix = "► " if percent == max_prob else "  "
                detail_text += f"{prefix}{clean_name:<15} : {percent:.1f}%\n"

            self.lbl_stats_detail.configure(text=detail_text.strip())
            self.lbl_rgb.configure(text=f"R:{features[0]:.0f}  G:{features[1]:.0f}  B:{features[2]:.0f}")
            self.lbl_shape.configure(text=f"Hu1: {features[3]:.4f}")

        except Exception as e:
            print(e)
            messagebox.showerror(
                "Error Analysis",
                f"Terjadi kesalahan saat memproses gambar.\nDetail: {e}"
            )


if __name__ == "__main__":
    app = RoseClassifierApp()
    app.mainloop()
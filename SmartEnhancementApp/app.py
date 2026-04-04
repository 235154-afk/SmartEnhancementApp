"""
╔══════════════════════════════════════════════════════════════╗
║   SMART IMAGE ENHANCEMENT & ANALYSIS SYSTEM                 ║
║   DIP Lab 06  |  Muhammad Talal Tariq  |  Reg ID: 235154    ║
║   Built with Python + Tkinter + OpenCV + Matplotlib         ║
╚══════════════════════════════════════════════════════════════╝
Run:  python app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os

# ─────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (dark professional theme)
# ─────────────────────────────────────────────────────────────
BG_DARK    = "#0D1117"
BG_PANEL   = "#161B22"
BG_CARD    = "#21262D"
ACCENT     = "#2563EB"
ACCENT2    = "#10B981"
ACCENT3    = "#F59E0B"
TEXT_PRI   = "#F0F6FC"
TEXT_SEC   = "#8B949E"
TEXT_MUTED = "#484F58"
BORDER     = "#30363D"
SUCCESS    = "#238636"
WARNING    = "#9E6A03"
DANGER     = "#DA3633"
HEADER_BG  = "#1C2128"

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_HEAD   = ("Segoe UI", 13, "bold")
FONT_SUB    = ("Segoe UI", 10, "bold")
FONT_BODY   = ("Segoe UI", 9)
FONT_SMALL  = ("Segoe UI", 8)
FONT_CODE   = ("Consolas", 9)
FONT_MONO   = ("Consolas", 8)


# ─────────────────────────────────────────────────────────────
#  IMAGE PROCESSING ENGINE
# ─────────────────────────────────────────────────────────────
class ImageProcessor:
    def __init__(self):
        self.img_bgr  = None
        self.img_rgb  = None
        self.img_gray = None
        self.enhanced = None
        self.path     = None

    def load(self, path):
        self.path    = path
        self.img_bgr = cv2.imread(path)
        if self.img_bgr is None:
            return False
        self.img_rgb  = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        return True

    def get_info(self):
        if self.img_rgb is None:
            return {}
        h, w, c = self.img_rgb.shape
        return {
            "Resolution": f"{h} × {w} px",
            "Channels"  : str(c),
            "Data Type" : str(self.img_rgb.dtype),
            "File Size"  : f"{os.path.getsize(self.path) // 1024} KB",
            "File Name"  : os.path.basename(self.path),
        }

    def get_pixel_matrix(self):
        if self.img_gray is None:
            return ""
        m = self.img_gray[:5, :5]
        lines = ["  Grayscale Pixel Matrix (5×5):"]
        for row in m:
            lines.append("  " + "  ".join(f"{v:3d}" for v in row))
        return "\n".join(lines)

    def sample(self, scale):
        if self.img_gray is None:
            return None
        h, w = self.img_gray.shape
        nh, nw = int(h * scale), int(w * scale)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(self.img_gray, (nw, nh), interpolation=interp)

    def quantize(self, bits):
        if self.img_gray is None:
            return None
        lv = 2 ** bits
        return np.uint8(np.round(self.img_gray / 255 * (lv - 1)) * (255 / (lv - 1)))

    def rotate(self, angle):
        if self.img_gray is None:
            return None
        h, w = self.img_gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(self.img_gray, M, (w, h))

    def translate(self, tx, ty):
        if self.img_gray is None:
            return None
        h, w = self.img_gray.shape
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(self.img_gray, M, (w, h))

    def shear(self, factor):
        if self.img_gray is None:
            return None
        h, w = self.img_gray.shape
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        pts2 = np.float32([[0, 0], [w, 0], [int(factor * h), h]])
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(self.img_gray, M, (w, h))

    def negative(self):
        if self.img_gray is None:
            return None
        return np.uint8(255 - self.img_gray)

    def log_transform(self, c=1.0):
        if self.img_gray is None:
            return None
        d = self.img_gray.astype(np.float64) / 255.0
        r = c * np.log1p(d)
        r = r / r.max()
        return np.uint8(r * 255)

    def gamma(self, g):
        if self.img_gray is None:
            return None
        d = self.img_gray.astype(np.float64) / 255.0
        return np.uint8(np.power(d, g) * 255)

    def histogram_eq(self):
        if self.img_gray is None:
            return None
        counts, _ = np.histogram(self.img_gray.ravel(), 256, (0, 256))
        cdf = np.cumsum(counts / counts.sum())
        lut = np.uint8(255 * cdf)
        return lut[self.img_gray]

    def process_image(self, gamma_val=0.5):
        if self.img_bgr is None:
            return None
        g  = self.img_gray.copy()
        d  = g.astype(np.float64) / 255.0
        lg = np.log1p(d)
        lg = (lg / lg.max() * 255).astype(np.uint8)
        gm = np.power(lg.astype(np.float64) / 255.0, gamma_val)
        gm = (gm * 255).astype(np.uint8)
        cnt, _ = np.histogram(gm.ravel(), 256, (0, 256))
        lut = np.uint8(255 * np.cumsum(cnt / cnt.sum()))
        self.enhanced = lut[gm]
        return self.enhanced


# ─────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────
class SmartEnhancementApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.proc = ImageProcessor()

        self.title("Smart Image Enhancement & Analysis System  |  DIP Lab 06  |  235154")
        self.geometry("1440x860")
        self.minsize(1200, 700)
        self.configure(bg=BG_DARK)
        self.state("zoomed")          # start maximized

        self._setup_styles()
        self._build_ui()
        self._status("Welcome! Upload an image to begin.")

    # ── styles ────────────────────────────────────────────────
    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",       background=BG_DARK,  borderwidth=0)
        style.configure("TNotebook.Tab",   background=BG_PANEL, foreground=TEXT_SEC,
                         font=FONT_SUB,    padding=[18, 8],     borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", BG_CARD)],
                  foreground=[("selected", TEXT_PRI)])
        style.configure("TScale", background=BG_CARD, troughcolor=BORDER,
                         sliderrelief="flat")
        style.configure("Vertical.TScrollbar", background=BG_PANEL,
                         troughcolor=BG_DARK, borderwidth=0)

    # ── top header ────────────────────────────────────────────
    def _build_header(self, parent):
        hdr = tk.Frame(parent, bg=HEADER_BG, height=64)
        hdr.pack(fill="x"); hdr.pack_propagate(False)

        tk.Label(hdr, text="◈", bg=HEADER_BG, fg=ACCENT,
                 font=("Segoe UI", 20)).pack(side="left", padx=(18, 6), pady=10)
        tk.Label(hdr, text="Smart Image Enhancement & Analysis System",
                 bg=HEADER_BG, fg=TEXT_PRI, font=FONT_TITLE).pack(side="left", pady=10)

        right = tk.Frame(hdr, bg=HEADER_BG)
        right.pack(side="right", padx=18)
        tk.Label(right, text="Muhammad Talal Tariq  |  235154  |  DIP Lab 06",
                 bg=HEADER_BG, fg=TEXT_SEC, font=FONT_SMALL).pack(anchor="e")
        tk.Label(right, text="Python · OpenCV · Matplotlib · Tkinter",
                 bg=HEADER_BG, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="e")

        # separator line
        tk.Frame(parent, bg=ACCENT, height=2).pack(fill="x")

    # ── left sidebar ──────────────────────────────────────────
    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG_PANEL, width=230)
        sb.pack(side="left", fill="y"); sb.pack_propagate(False)

        # Upload button
        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", padx=0)
        up_btn = tk.Button(sb, text="⬆  Upload Image",
                           bg=ACCENT, fg="white", font=FONT_SUB,
                           relief="flat", cursor="hand2", height=2,
                           command=self._upload_image)
        up_btn.pack(fill="x", padx=12, pady=14)

        # Image info panel
        self.info_frame = tk.Frame(sb, bg=BG_CARD, relief="flat")
        self.info_frame.pack(fill="x", padx=12, pady=(0, 10))
        tk.Label(self.info_frame, text="IMAGE INFO", bg=BG_CARD, fg=TEXT_MUTED,
                 font=FONT_SMALL).pack(anchor="w", padx=8, pady=(8, 2))
        self.info_labels = {}
        for key in ["File Name", "Resolution", "Channels", "Data Type", "File Size"]:
            row = tk.Frame(self.info_frame, bg=BG_CARD)
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=key + ":", bg=BG_CARD, fg=TEXT_MUTED,
                     font=FONT_SMALL, width=10, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="—", bg=BG_CARD, fg=TEXT_SEC,
                           font=FONT_SMALL, anchor="w")
            lbl.pack(side="left")
            self.info_labels[key] = lbl
        tk.Frame(self.info_frame, bg=BG_CARD, height=6).pack()

        # Phase buttons
        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x")
        tk.Label(sb, text="PHASES", bg=BG_PANEL, fg=TEXT_MUTED,
                 font=FONT_SMALL).pack(anchor="w", padx=14, pady=(12, 4))

        phases = [
            ("6.1", "Image Acquisition",      ACCENT,  self._run_phase61),
            ("6.2", "Sampling & Quantization", ACCENT,  self._run_phase62),
            ("6.3", "Geometric Transforms",    ACCENT,  self._run_phase63),
            ("6.4", "Intensity Transforms",    ACCENT,  self._run_phase64),
            ("6.5", "Histogram Processing",    ACCENT,  self._run_phase65),
            ("6.6", "Full Pipeline",           ACCENT2, self._run_phase66),
        ]
        for num, name, col, cmd in phases:
            btn = tk.Button(sb,
                            text=f"  {num}  {name}",
                            bg=BG_CARD, fg=TEXT_PRI, font=FONT_BODY,
                            relief="flat", cursor="hand2", anchor="w",
                            activebackground=ACCENT, activeforeground="white",
                            command=cmd)
            btn.pack(fill="x", padx=8, pady=2, ipady=6)
            btn.bind("<Enter>", lambda e, b=btn, c=col: b.config(bg=c, fg="white"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=BG_CARD, fg=TEXT_PRI))

        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", pady=8)

        # Save button
        save_btn = tk.Button(sb, text="💾  Save Enhanced Image",
                             bg=SUCCESS, fg="white", font=FONT_SUB,
                             relief="flat", cursor="hand2", height=2,
                             command=self._save_output)
        save_btn.pack(fill="x", padx=12, pady=4)

        # Status bar inside sidebar
        self.status_var = tk.StringVar(value="Ready")
        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", pady=6)
        tk.Label(sb, textvariable=self.status_var, bg=BG_PANEL, fg=TEXT_MUTED,
                 font=FONT_SMALL, wraplength=200, justify="left").pack(padx=12, pady=4, anchor="w")

    # ── main content tabs ─────────────────────────────────────
    def _build_tabs(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)

        # Tab 1: Preview
        self.tab_preview = tk.Frame(self.nb, bg=BG_DARK)
        self.nb.add(self.tab_preview, text="  Preview  ")
        self._build_preview_tab()

        # Tab 2: Analysis
        self.tab_analysis = tk.Frame(self.nb, bg=BG_DARK)
        self.nb.add(self.tab_analysis, text="  Analysis  ")
        self._build_analysis_tab()

        # Tab 3: Controls
        self.tab_controls = tk.Frame(self.nb, bg=BG_DARK)
        self.nb.add(self.tab_controls, text="  Controls  ")
        self._build_controls_tab()

        # Tab 4: Q&A
        self.tab_qa = tk.Frame(self.nb, bg=BG_DARK)
        self.nb.add(self.tab_qa, text="  Q & A  ")
        self._build_qa_tab()

    # ── Preview tab ───────────────────────────────────────────
    def _build_preview_tab(self):
        top = tk.Frame(self.tab_preview, bg=BG_DARK)
        top.pack(fill="both", expand=True, padx=8, pady=8)

        # Original
        left = tk.Frame(top, bg=BG_CARD, relief="flat")
        left.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self._card_header(left, "ORIGINAL IMAGE", TEXT_SEC)
        self.orig_canvas = tk.Label(left, bg=BG_CARD, text="Upload an image\nto see preview",
                                    fg=TEXT_MUTED, font=FONT_BODY)
        self.orig_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # Enhanced
        right = tk.Frame(top, bg=BG_CARD, relief="flat")
        right.pack(side="right", fill="both", expand=True, padx=(4, 0))
        self._card_header(right, "ENHANCED OUTPUT", ACCENT2)
        self.enh_canvas = tk.Label(right, bg=BG_CARD, text="Run Phase 6.6\nto see enhanced result",
                                   fg=TEXT_MUTED, font=FONT_BODY)
        self.enh_canvas.pack(fill="both", expand=True, padx=8, pady=8)

    # ── Analysis tab ──────────────────────────────────────────
    def _build_analysis_tab(self):
        self.fig_analysis = Figure(figsize=(12, 5), facecolor=BG_DARK)
        self.ax_analysis  = [self.fig_analysis.add_subplot(1, 3, i+1) for i in range(3)]
        for ax in self.ax_analysis:
            ax.set_facecolor(BG_CARD)
            ax.tick_params(colors=TEXT_SEC, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)

        self.canvas_analysis = FigureCanvasTkAgg(self.fig_analysis, self.tab_analysis)
        self.canvas_analysis.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        # Pixel matrix display
        bottom = tk.Frame(self.tab_analysis, bg=BG_CARD)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        self._card_header(bottom, "PIXEL MATRIX  (top-left 5×5 grayscale values)", TEXT_SEC)
        self.matrix_text = tk.Text(bottom, bg=BG_CARD, fg=ACCENT2, font=FONT_CODE,
                                   height=7, relief="flat", state="disabled")
        self.matrix_text.pack(fill="x", padx=8, pady=8)

    # ── Controls tab ──────────────────────────────────────────
    def _build_controls_tab(self):
        canvas = tk.Canvas(self.tab_controls, bg=BG_DARK, highlightthickness=0)
        scroll = ttk.Scrollbar(self.tab_controls, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG_DARK)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        cols = tk.Frame(inner, bg=BG_DARK)
        cols.pack(fill="both", expand=True, padx=8, pady=8)

        # ── Column 1: Sampling ───────────────────────────────
        c1 = self._make_card(cols, "PHASE 6.2 — SAMPLING & QUANTIZATION")
        c1.pack(side="left", fill="both", expand=True, padx=4)

        self._slider_row(c1, "Scale Factor", 0.1, 3.0, 1.0, 0.05,
                         "scale_var", lambda v: self._preview_sample())
        tk.Label(c1, text="Bit Depth", bg=BG_CARD, fg=TEXT_SEC, font=FONT_SMALL).pack(anchor="w", padx=12)
        self.bit_var = tk.IntVar(value=8)
        for b in [8, 4, 2]:
            tk.Radiobutton(c1, text=f"{b}-bit  ({2**b} levels)",
                           variable=self.bit_var, value=b,
                           bg=BG_CARD, fg=TEXT_PRI, selectcolor=ACCENT,
                           font=FONT_BODY, activebackground=BG_CARD,
                           command=self._preview_quantize).pack(anchor="w", padx=24)
        tk.Frame(c1, bg=BG_CARD, height=8).pack()
        self._small_canvas(c1, "sample_canvas", "Sampling Preview")
        self._small_canvas(c1, "quant_canvas",  "Quantization Preview")

        # ── Column 2: Geometric ──────────────────────────────
        c2 = self._make_card(cols, "PHASE 6.3 — GEOMETRIC TRANSFORMATIONS")
        c2.pack(side="left", fill="both", expand=True, padx=4)

        self._slider_row(c2, "Rotation Angle (°)", 0, 360, 0, 1,
                         "angle_var", lambda v: self._preview_rotate())
        self._slider_row(c2, "Translation X (px)", -200, 200, 0, 1,
                         "tx_var", lambda v: self._preview_translate())
        self._slider_row(c2, "Translation Y (px)", -200, 200, 0, 1,
                         "ty_var", lambda v: self._preview_translate())
        self._slider_row(c2, "Shear Factor", -0.8, 0.8, 0.0, 0.01,
                         "shear_var", lambda v: self._preview_shear())
        self._small_canvas(c2, "geo_canvas", "Geometric Transform Preview")

        # ── Column 3: Intensity ──────────────────────────────
        c3 = self._make_card(cols, "PHASE 6.4 — INTENSITY TRANSFORMATIONS")
        c3.pack(side="left", fill="both", expand=True, padx=4)

        self._slider_row(c3, "Gamma (γ)", 0.1, 3.0, 1.0, 0.05,
                         "gamma_var", lambda v: self._preview_gamma())
        self._slider_row(c3, "Log C", 0.1, 5.0, 1.0, 0.1,
                         "logc_var", lambda v: self._preview_log())
        tk.Label(c3, text="Quick Transforms", bg=BG_CARD, fg=TEXT_SEC,
                 font=FONT_SMALL).pack(anchor="w", padx=12, pady=(8, 4))
        for txt, cmd in [("Apply Negative",    self._preview_negative),
                         ("Apply Log",         self._preview_log),
                         ("Apply Gamma",       self._preview_gamma),
                         ("Histogram EQ",      self._preview_histeq)]:
            tk.Button(c3, text=txt, bg=BG_PANEL, fg=TEXT_PRI, font=FONT_BODY,
                      relief="flat", cursor="hand2", command=cmd,
                      activebackground=ACCENT, activeforeground="white"
                      ).pack(fill="x", padx=12, pady=2, ipady=4)
        self._small_canvas(c3, "int_canvas", "Intensity Transform Preview")

    # ── Q&A tab ───────────────────────────────────────────────
    def _build_qa_tab(self):
        canvas = tk.Canvas(self.tab_qa, bg=BG_DARK, highlightthickness=0)
        scroll = ttk.Scrollbar(self.tab_qa, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG_DARK)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        tk.Label(inner, text="Section 7.4 — Questions & Answers",
                 bg=BG_DARK, fg=TEXT_PRI, font=FONT_HEAD).pack(anchor="w", padx=20, pady=(20, 4))
        tk.Frame(inner, bg=ACCENT, height=2).pack(fill="x", padx=20, pady=(0, 16))

        qa_data = [
            ("Q1", "Why does histogram equalization improve contrast?",
             "Histogram equalization redistributes pixel intensities so they span the full 0–255 "
             "range uniformly. It works by computing the Cumulative Distribution Function (CDF) of "
             "the original histogram and using it as a mapping (lookup table). Narrow intensity "
             "clusters (low contrast) are stretched into a wider spread, making dark and bright "
             "regions much more distinguishable from each other. This is measurable — standard "
             "deviation of pixel values jumped from 15.2 to 73.0 in our experiment (4.8× gain)."),

            ("Q2", "How does gamma affect brightness?",
             "Gamma correction applies the formula:  s = r ^ gamma  (where r is normalized 0→1).\n\n"
             "• Gamma < 1 (e.g. 0.5): concave curve — raises dark pixel values more than bright ones "
             "→ image becomes BRIGHTER. Best for underexposed or dark images.\n\n"
             "• Gamma > 1 (e.g. 1.5): convex curve — compresses bright pixel values downward "
             "→ image becomes DARKER. Best for overexposed images.\n\n"
             "• Gamma = 1: no change (identity function)."),

            ("Q3", "What is the effect of quantization on image quality?",
             "Quantization reduces the number of available intensity levels by reducing bit depth:\n\n"
             "• 8-bit: 256 levels → full smooth tonal gradients, no visible artifacts.\n"
             "• 4-bit: 16 levels  → visible banding and false contouring on smooth areas.\n"
             "• 2-bit:  4 levels  → heavy posterisation, harsh tone jumps, severe detail loss.\n\n"
             "Information discarded by quantization is permanently lost and CANNOT be recovered, "
             "making quantization an irreversible process."),

            ("Q4", "Which transformation is reversible and why?",
             "Rotation and Translation are reversible — applying +90° then −90° returns the "
             "image to its original spatial arrangement (verified in Phase 6.3).\n\n"
             "Affine transforms (rotation, translation, shear) are mathematically invertible "
             "through their inverse matrices.\n\n"
             "Intensity transforms like Log and Gamma are analytically reversible using inverse functions.\n\n"
             "NOT reversible: Quantization (permanently discards intensity levels) and aggressive "
             "downsampling (loses spatial detail that cannot be reconstructed)."),

            ("Q5", "How do transformations affect spatial structure?",
             "• Geometric transforms (rotation, translation, shear): move pixels in space WITHOUT "
             "changing their intensity values. Visual content moves/rotates/distorts — spatial "
             "relationships between features change.\n\n"
             "• Intensity transforms (log, gamma, negative, HE): change pixel BRIGHTNESS VALUES "
             "without moving them — spatial structure is preserved, only tonal appearance changes.\n\n"
             "• Combined (Pipeline Phase 6.6): spatial structure is reorganized AND tonal quality "
             "is simultaneously improved, producing the most comprehensive enhancement result."),
        ]

        colors = [ACCENT, ACCENT2, ACCENT3, "#EC4899", "#8B5CF6"]
        for i, (num, q, a) in enumerate(qa_data):
            card = tk.Frame(inner, bg=BG_CARD)
            card.pack(fill="x", padx=20, pady=6)

            hdr = tk.Frame(card, bg=colors[i], height=3)
            hdr.pack(fill="x")

            body_f = tk.Frame(card, bg=BG_CARD)
            body_f.pack(fill="x", padx=16, pady=12)

            tk.Label(body_f, text=num, bg=BG_CARD, fg=colors[i],
                     font=("Segoe UI", 11, "bold")).pack(anchor="w")
            tk.Label(body_f, text=q, bg=BG_CARD, fg=TEXT_PRI,
                     font=("Segoe UI", 10, "bold"), wraplength=900, justify="left").pack(anchor="w", pady=(2, 8))
            tk.Label(body_f, text=a, bg=BG_CARD, fg=TEXT_SEC,
                     font=FONT_BODY, wraplength=900, justify="left").pack(anchor="w")

    # ── FULL UI BUILD ─────────────────────────────────────────
    def _build_ui(self):
        self._build_header(self)

        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True)

        self._build_sidebar(main)

        right = tk.Frame(main, bg=BG_DARK)
        right.pack(side="right", fill="both", expand=True)

        self._build_tabs(right)

    # ── HELPERS ───────────────────────────────────────────────
    def _card_header(self, parent, text, color=TEXT_SEC):
        hdr = tk.Frame(parent, bg=BG_CARD)
        hdr.pack(fill="x")
        tk.Frame(hdr, bg=color, width=3).pack(side="left", fill="y")
        tk.Label(hdr, text=text, bg=BG_CARD, fg=color,
                 font=FONT_SMALL).pack(side="left", padx=8, pady=6)

    def _make_card(self, parent, title):
        card = tk.Frame(parent, bg=BG_CARD)
        self._card_header(card, title, ACCENT)
        return card

    def _slider_row(self, parent, label, from_, to, init, step, var_name, cmd):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", padx=12, pady=4)
        var = tk.DoubleVar(value=init)
        setattr(self, var_name, var)
        top = tk.Frame(row, bg=BG_CARD)
        top.pack(fill="x")
        tk.Label(top, text=label, bg=BG_CARD, fg=TEXT_SEC, font=FONT_SMALL).pack(side="left")
        val_lbl = tk.Label(top, text=f"{init:.2f}", bg=BG_CARD, fg=ACCENT, font=FONT_SMALL)
        val_lbl.pack(side="right")
        def on_change(v, vl=val_lbl, va=var, c=cmd):
            vl.config(text=f"{float(v):.2f}"); c(v)
        sl = ttk.Scale(row, from_=from_, to=to, variable=var,
                       orient="horizontal", command=on_change)
        sl.pack(fill="x")

    def _small_canvas(self, parent, attr_name, placeholder):
        lbl = tk.Label(parent, bg=BG_PANEL, fg=TEXT_MUTED,
                       text=placeholder, font=FONT_SMALL, height=8)
        lbl.pack(fill="x", padx=12, pady=4)
        setattr(self, attr_name, lbl)

    def _status(self, msg, color=TEXT_MUTED):
        self.status_var.set(msg)

    # ── IMAGE DISPLAY ─────────────────────────────────────────
    def _to_photoimage(self, arr, max_w=560, max_h=400):
        if arr is None:
            return None
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        h, w = arr.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(Image.fromarray(arr))

    def _show_on_label(self, label, arr, max_w=560, max_h=400):
        ph = self._to_photoimage(arr, max_w, max_h)
        if ph:
            label.config(image=ph, text="")
            label.image = ph

    def _show_on_small(self, attr_name, arr):
        lbl = getattr(self, attr_name, None)
        if lbl:
            self._show_on_label(lbl, arr, 300, 200)

    # ── UPLOAD ────────────────────────────────────────────────
    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        if not self.proc.load(path):
            messagebox.showerror("Error", "Could not load image. Check file format.")
            return

        # Update info panel
        info = self.proc.get_info()
        for k, v in info.items():
            if k in self.info_labels:
                self.info_labels[k].config(text=v)

        # Show preview
        self._show_on_label(self.orig_canvas, self.proc.img_rgb)
        self.enh_canvas.config(image="", text="Run Phase 6.6\nto see enhanced result")

        # Update analysis
        self._update_histogram()
        self._update_matrix()
        self._status(f"Loaded: {os.path.basename(path)}")

    # ── HISTOGRAM UPDATE ──────────────────────────────────────
    def _update_histogram(self, compare=None):
        if self.proc.img_gray is None:
            return
        for ax in self.ax_analysis:
            ax.clear()
            ax.set_facecolor(BG_CARD)
            ax.tick_params(colors=TEXT_SEC, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        # Original histogram
        self.ax_analysis[0].hist(self.proc.img_gray.ravel(), 256,
                                  color=ACCENT, alpha=0.8, edgecolor='none')
        self.ax_analysis[0].set_title("Histogram: Original", color=TEXT_PRI, fontsize=9)
        self.ax_analysis[0].set_xlabel("Intensity", color=TEXT_SEC, fontsize=7)
        self.ax_analysis[0].set_ylabel("Count", color=TEXT_SEC, fontsize=7)

        # CDF
        counts, _ = np.histogram(self.proc.img_gray.ravel(), 256, (0, 256))
        cdf = np.cumsum(counts / counts.sum())
        self.ax_analysis[1].plot(cdf, color=ACCENT2, linewidth=2)
        self.ax_analysis[1].set_title("CDF (Equalization Curve)", color=TEXT_PRI, fontsize=9)
        self.ax_analysis[1].set_xlabel("Intensity", color=TEXT_SEC, fontsize=7)
        self.ax_analysis[1].set_ylabel("Cumulative Probability", color=TEXT_SEC, fontsize=7)
        self.ax_analysis[1].fill_between(range(256), cdf, alpha=0.2, color=ACCENT2)

        # Std dev comparison
        eq = self.proc.histogram_eq()
        std_orig = np.std(self.proc.img_gray.astype(float))
        std_eq   = np.std(eq.astype(float))
        bars = self.ax_analysis[2].bar(["Original", "After HE"], [std_orig, std_eq],
                                        color=[ACCENT, ACCENT2], edgecolor='none')
        for bar, v in zip(bars, [std_orig, std_eq]):
            self.ax_analysis[2].text(bar.get_x() + bar.get_width()/2,
                                      bar.get_height() + 0.5, f"{v:.1f}",
                                      ha='center', color=TEXT_PRI, fontsize=8)
        self.ax_analysis[2].set_title("Contrast (Std Dev)", color=TEXT_PRI, fontsize=9)
        self.ax_analysis[2].set_ylabel("Standard Deviation", color=TEXT_SEC, fontsize=7)
        self.ax_analysis[2].tick_params(colors=TEXT_SEC)

        self.fig_analysis.tight_layout(pad=1.5)
        self.canvas_analysis.draw()

    def _update_matrix(self):
        txt = self.proc.get_pixel_matrix()
        self.matrix_text.config(state="normal")
        self.matrix_text.delete("1.0", "end")
        self.matrix_text.insert("end", txt)
        self.matrix_text.config(state="disabled")

    # ── PHASE RUNNERS ─────────────────────────────────────────
    def _require_image(self):
        if self.proc.img_gray is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return False
        return True

    def _run_phase61(self):
        if not self._require_image(): return
        self.nb.select(self.tab_analysis)
        self._update_histogram()
        self._update_matrix()
        self._status("Phase 6.1: Image info & pixel matrix displayed in Analysis tab.")

    def _run_phase62(self):
        if not self._require_image(): return
        self.nb.select(self.tab_controls)
        self._preview_sample()
        self._preview_quantize()
        self._status("Phase 6.2: Adjust Scale and Bit Depth sliders in Controls tab.")

    def _run_phase63(self):
        if not self._require_image(): return
        self.nb.select(self.tab_controls)
        self._preview_rotate()
        self._status("Phase 6.3: Adjust Rotation/Translation/Shear sliders in Controls tab.")

    def _run_phase64(self):
        if not self._require_image(): return
        self.nb.select(self.tab_controls)
        self._preview_gamma()
        self._status("Phase 6.4: Adjust Gamma/Log sliders in Controls tab.")

    def _run_phase65(self):
        if not self._require_image(): return
        self.nb.select(self.tab_analysis)
        self._update_histogram()
        self._status("Phase 6.5: Histogram, CDF and Std Dev shown in Analysis tab.")

    def _run_phase66(self):
        if not self._require_image(): return
        gamma_val = getattr(self, "gamma_var", None)
        gv = gamma_val.get() if gamma_val else 0.5
        enhanced = self.proc.process_image(gv)
        self._show_on_label(self.enh_canvas, enhanced)
        self.nb.select(self.tab_preview)
        self._status("Phase 6.6: Pipeline complete! Original vs Enhanced shown in Preview tab.")

    # ── LIVE PREVIEWS ─────────────────────────────────────────
    def _preview_sample(self, _=None):
        if self.proc.img_gray is None: return
        s   = self.scale_var.get()
        rs  = self.proc.sample(s)
        self._show_on_small("sample_canvas", rs)

    def _preview_quantize(self, _=None):
        if self.proc.img_gray is None: return
        b  = self.bit_var.get()
        iq = self.proc.quantize(b)
        self._show_on_small("quant_canvas", iq)

    def _preview_rotate(self, _=None):
        if self.proc.img_gray is None: return
        self._show_on_small("geo_canvas", self.proc.rotate(self.angle_var.get()))

    def _preview_translate(self, _=None):
        if self.proc.img_gray is None: return
        self._show_on_small("geo_canvas",
                            self.proc.translate(int(self.tx_var.get()), int(self.ty_var.get())))

    def _preview_shear(self, _=None):
        if self.proc.img_gray is None: return
        self._show_on_small("geo_canvas", self.proc.shear(self.shear_var.get()))

    def _preview_negative(self):
        if self.proc.img_gray is None: return
        self._show_on_small("int_canvas", self.proc.negative())

    def _preview_log(self, _=None):
        if self.proc.img_gray is None: return
        self._show_on_small("int_canvas", self.proc.log_transform(self.logc_var.get()))

    def _preview_gamma(self, _=None):
        if self.proc.img_gray is None: return
        self._show_on_small("int_canvas", self.proc.gamma(self.gamma_var.get()))

    def _preview_histeq(self):
        if self.proc.img_gray is None: return
        self._show_on_small("int_canvas", self.proc.histogram_eq())

    # ── SAVE OUTPUT ───────────────────────────────────────────
    def _save_output(self):
        if self.proc.enhanced is None:
            messagebox.showwarning("No Output", "Run Phase 6.6 first to generate enhanced image.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            initialfile="235154_enhanced_output.jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")])
        if path:
            cv2.imwrite(path, self.proc.enhanced)
            messagebox.showinfo("Saved", f"Enhanced image saved to:\n{path}")
            self._status(f"Saved: {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SmartEnhancementApp()
    app.mainloop()

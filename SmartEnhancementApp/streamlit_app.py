"""
Smart Image Enhancement & Analysis System
DIP|Muhammad Talal Tariq|235154
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Image Enhancement | DIP ",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0D1117; }
    .block-container { padding-top: 1rem; }
    h1 { color: #2563EB !important; }
    h2, h3 { color: #F0F6FC !important; }
    .stButton>button {
        background-color: #2563EB; color: white;
        border: none; border-radius: 6px; font-weight: 600;
        padding: 0.5rem 1.5rem; width: 100%;
    }
    .stButton>button:hover { background-color: #1D4ED8; }
    .metric-card {
        background: #21262D; border-radius: 8px;
        padding: 12px 16px; margin: 4px 0;
        border-left: 3px solid #2563EB;
    }
    .qa-card {
        background: #161B22; border-radius: 8px;
        padding: 16px; margin: 8px 0;
        border-left: 4px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("# ◈ Smart Image Enhancement & Analysis System")
st.markdown("**DIP ** | Muhammad Talal Tariq | 235154 | Python · OpenCV ")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⬆ Upload Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp"])
    st.markdown("---")
    st.markdown("### ⚙ Controls")
    gamma_val = st.slider("Gamma (γ)", 0.1, 3.0, 0.5, 0.05)
    scale_val = st.slider("Scale Factor", 0.1, 3.0, 1.0, 0.05)
    bits_val  = st.radio("Bit Depth", [8, 4, 2], index=0)
    angle_val = st.slider("Rotation Angle (°)", 0, 360, 0, 1)
    tx_val    = st.slider("Translation X", -200, 200, 0, 1)
    ty_val    = st.slider("Translation Y", -200, 200, 0, 1)
    shear_val = st.slider("Shear Factor", -0.8, 0.8, 0.0, 0.01)
    st.markdown("---")
    st.markdown("**Muhammad Talal Tariq**")
    st.markdown("Reg ID: 235154 | DIP Lab 06")

# ── Load image ────────────────────────────────────────────────
def load_image(upload):
    arr = np.frombuffer(upload.read(), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, rgb, gray

# ── Tabs ──────────────────────────────────────────────────────
tabs = st.tabs(["📷 Preview", "📊 Analysis", "🔧 Transforms", "❓ Q & A"])

if uploaded:
    bgr, rgb, gray = load_image(uploaded)
    h, w, c = rgb.shape

    # ── Tab 1: Preview ───────────────────────────────────────
    with tabs[0]:
        st.markdown("## Phase 6.1 — Image Acquisition & Understanding")
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb, caption="Original RGB Image", use_container_width=True)
        with col2:
            st.image(gray, caption="Grayscale Image", clamp=True, use_container_width=True)

        st.markdown("### Image Report")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Resolution", f"{h}×{w}")
        c2.metric("Channels", str(c))
        c3.metric("Data Type", str(rgb.dtype))
        c4.metric("Total Pixels", f"{h*w:,}")
        c5.metric("File Name", uploaded.name[:12])

        st.markdown("### Pixel Matrix (top-left 5×5 grayscale)")
        st.code(str(gray[:5, :5]), language="python")

        # Phase 6.6 pipeline
        st.markdown("---")
        st.markdown("## Phase 6.6 — Final Enhanced Output")
        d  = gray.astype(np.float64)/255.0
        lg = np.log1p(d); lg = (lg/lg.max()*255).astype(np.uint8)
        gm = np.power(lg.astype(np.float64)/255.0, gamma_val)
        gm = (gm*255).astype(np.uint8)
        cnt, _ = np.histogram(gm.ravel(), 256, (0,256))
        lut = np.uint8(255*np.cumsum(cnt/cnt.sum()))
        enhanced = lut[gm]

        col1, col2 = st.columns(2)
        with col1:
            st.image(gray, caption="Before — Grayscale", clamp=True, use_container_width=True)
        with col2:
            st.image(enhanced, caption=f"After — Enhanced (γ={gamma_val})", clamp=True, use_container_width=True)

        buf = io.BytesIO()
        Image.fromarray(enhanced).save(buf, format="JPEG")
        st.download_button("💾 Download Enhanced Image", buf.getvalue(),
                           "235154_enhanced_output.jpg", "image/jpeg")

    # ── Tab 2: Analysis ──────────────────────────────────────
    with tabs[1]:
        st.markdown("## Phase 6.2 — Sampling & Phase 6.5 — Histogram")

        # Sampling
        st.markdown("### Sampling Comparison")
        scales = [0.25, 0.5, 1.0, 1.5, 2.0]
        cols = st.columns(5)
        for i, s in enumerate(scales):
            nh, nw = int(h*s), int(w*s)
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
            rs = cv2.resize(gray, (nw, nh), interpolation=interp)
            cols[i].image(rs, caption=f"{s}×\n{nh}×{nw}", clamp=True, use_container_width=True)

        # Bit depth
        st.markdown("### Quantization (Bit Depth)")
        cols = st.columns(3)
        for i, b in enumerate([8, 4, 2]):
            lv = 2**b
            iq = np.uint8(np.round(gray/255*(lv-1))*(255/(lv-1)))
            cols[i].image(iq, caption=f"{b}-bit ({lv} levels)", clamp=True, use_container_width=True)

        # Histograms
        st.markdown("### Phase 6.5 — Histogram Processing")
        counts, _ = np.histogram(gray.ravel(), 256, (0,256))
        pdf = counts/counts.sum(); cdf = np.cumsum(pdf)
        lut2 = np.uint8(255*cdf); eq = lut2[gray]
        eq_cv = cv2.equalizeHist(gray)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#0D1117")
        for ax, im, t, col in zip(axes,
                                   [gray, eq, eq_cv],
                                   ["Original","Manual HE","cv2 HE"],
                                   ["#2563EB","#10B981","#F59E0B"]):
            ax.hist(im.ravel(), 256, color=col, alpha=0.8, edgecolor='none')
            ax.set_title(t, color="white", fontsize=10)
            ax.set_facecolor("#21262D")
            ax.tick_params(colors="#8B949E")
            for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
        fig.tight_layout()
        st.pyplot(fig)

        std_o = np.std(gray.astype(float))
        std_m = np.std(eq.astype(float))
        c1, c2, c3 = st.columns(3)
        c1.metric("Std Dev — Original", f"{std_o:.1f}")
        c2.metric("Std Dev — Manual HE", f"{std_m:.1f}", f"+{std_m-std_o:.1f}")
        c3.metric("Improvement", f"{std_m/std_o:.1f}×")

        # CDF
        fig2, ax2 = plt.subplots(figsize=(8, 3), facecolor="#0D1117")
        ax2.plot(cdf, color="#10B981", linewidth=2)
        ax2.fill_between(range(256), cdf, alpha=0.2, color="#10B981")
        ax2.set_title("CDF Used for Equalization", color="white")
        ax2.set_facecolor("#21262D")
        ax2.tick_params(colors="#8B949E")
        for sp in ax2.spines.values(): sp.set_edgecolor("#30363D")
        st.pyplot(fig2)

    # ── Tab 3: Transforms ────────────────────────────────────
    with tabs[2]:
        st.markdown("## Phase 6.3 — Geometric Transformations")

        # Rotations
        st.markdown("### Rotation at 7 Angles")
        angles = [30, 45, 60, 90, 120, 150, 180]
        cols = st.columns(7)
        cx, cy = w//2, h//2
        for i, ang in enumerate(angles):
            M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
            rot = cv2.warpAffine(gray, M, (w,h))
            cols[i].image(rot, caption=f"{ang}°", clamp=True, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Translation")
            Mt = np.float32([[1,0,tx_val],[0,1,ty_val]])
            img_t = cv2.warpAffine(gray, Mt, (w,h))
            st.image(img_t, caption=f"tx={tx_val}, ty={ty_val}", clamp=True, use_container_width=True)
        with col2:
            st.markdown("### Shearing")
            pts1 = np.float32([[0,0],[w,0],[0,h]])
            pts2 = np.float32([[0,0],[w,0],[int(shear_val*h),h]])
            Ms = cv2.getAffineTransform(pts1,pts2)
            img_s = cv2.warpAffine(gray, Ms, (w,h))
            st.image(img_s, caption=f"Shear={shear_val}", clamp=True, use_container_width=True)
        with col3:
            st.markdown("### Inverse (Restore)")
            Mf = cv2.getRotationMatrix2D((cx,cy), angle_val, 1.0)
            Mi = cv2.getRotationMatrix2D((cx,cy), -angle_val, 1.0)
            r  = cv2.warpAffine(gray, Mf, (w,h))
            rs = cv2.warpAffine(r, Mi, (w,h))
            st.image(rs, caption="Restored", clamp=True, use_container_width=True)

        st.markdown("## Phase 6.4 — Intensity Transformations")
        d = gray.astype(np.float64)/255.0
        transforms = {
            "Original":     gray,
            "Negative":     np.uint8((1.0-d)*255),
            "Log (C=1)":    np.uint8(np.log1p(d)/np.log1p(d).max()*255),
            f"Gamma={gamma_val}": np.uint8(np.power(d, gamma_val)*255),
            "Histogram EQ": cv2.equalizeHist(gray),
        }
        cols = st.columns(5)
        for col, (name, img) in zip(cols, transforms.items()):
            col.image(img, caption=name, clamp=True, use_container_width=True)

    # ── Tab 4: Q&A ───────────────────────────────────────────
    with tabs[3]:
        st.markdown("## Section 7.4 — Questions & Answers")
        qa = [
            ("Q1", "Why does histogram equalization improve contrast?",
             "Histogram equalization redistributes pixel intensities across the full 0–255 range using the Cumulative Distribution Function (CDF) as a mapping function. Narrow intensity clusters are stretched into a wider spread, making dark and bright regions more distinguishable. In our experiment, std dev jumped from 15.2 → 73.0 (4.8× improvement)."),
            ("Q2", "How does gamma affect brightness?",
             "Formula: s = r^gamma\n• Gamma < 1 (e.g. 0.5): raises dark pixels → image BRIGHTER (underexposed fix)\n• Gamma > 1 (e.g. 1.5): compresses bright pixels → image DARKER (overexposed fix)\n• Gamma = 1: no change"),
            ("Q3", "What is the effect of quantization on image quality?",
             "8-bit (256 levels): full smooth quality\n4-bit (16 levels): visible banding/contouring\n2-bit (4 levels): heavy posterisation, severe detail loss\nInformation lost by quantization is PERMANENT — it cannot be recovered."),
            ("Q4", "Which transformation is reversible and why?",
             "Rotation & Translation are reversible via inverse matrices (+90° then -90° = original). Log and Gamma are analytically reversible. NOT reversible: quantization (permanently discards levels) and aggressive downsampling (loses spatial detail)."),
            ("Q5", "How do transformations affect spatial structure?",
             "Geometric transforms (rotation, translation, shear): move pixels in space WITHOUT changing values.\nIntensity transforms (log, gamma, HE): change pixel VALUES without moving them.\nCombined pipeline: reorganizes spatial structure AND improves tonal quality simultaneously."),
        ]
        colors = ["#2563EB","#10B981","#F59E0B","#EC4899","#8B5CF6"]
        for i, (num, q, a) in enumerate(qa):
            st.markdown(f"""
            <div class="qa-card" style="border-left-color:{colors[i]}">
                <strong style="color:{colors[i]}">{num}</strong>
                <p style="color:#F0F6FC;font-weight:600;margin:4px 0">{q}</p>
                <p style="color:#8B949E;white-space:pre-line">{a}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    for tab in tabs:
        with tab:
            st.info("⬆ Please upload an image using the sidebar to get started.")
            st.markdown("""
            ### How to use this app:
            1. Click **Browse files** in the sidebar
            2. Upload any image (JPG, PNG, BMP)
            3. Use the **Phase buttons** (sidebar sliders) to control transforms
            4. View results in each tab
            5. Download the enhanced image from the Preview tab
            """)

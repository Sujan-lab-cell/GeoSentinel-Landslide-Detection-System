import streamlit as st
from ultralytics import YOLO
from PIL import Image
import streamlit.components.v1 as components
import pyttsx3
import os
import cv2
import tempfile
import numpy as np
import folium
from streamlit_folium import st_folium
from twilio.rest import Client as TwilioClient
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64
from database import save_landslide
import pandas as pd
from folium.plugins import HeatMap
import uuid


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GeoSentinel · Landslide AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

model = YOLO("C:/Users/SUJAN K S/runs/segment/train21/weights/best.pt")

# ── Twilio ──────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = "AAC94f37bde61b10069660bd375e290690c"
TWILIO_AUTH_TOKEN  = "2ab0ee0d11a04043ca1ca95bd542abc5"
TWILIO_FROM_NUMBER = "+17752274509"
TWILIO_TO_NUMBER   ="+919108262847"

# ── SendGrid ─────────────────────────────────────────────────────────
SENDGRID_API_KEY   = "YOUR_SENDGRID_API_KEY"
ALERT_FROM_EMAIL   = "sujanksgowda2005@gmail.com"
ALERT_TO_EMAIL     = "nnm23am065@nmamit.in"


# ─────────────────────────────────────────────
# HELPER FUNCTIONS (original)
# ─────────────────────────────────────────────
def speak_alert(message):
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
    engine.setProperty('volume', 1.0)
    engine.save_to_file(message, "alert.mp3")
    engine.runAndWait()

def set_background(state: str):
    color_map = {"danger": "#1a0505", "safe": "#031a09", "none": "#0a0e1a"}
    bg = color_map.get(state, "#0a0e1a")
    components.html(f"""
        <script>
            (function() {{
                const app = window.parent.document.querySelector('.stApp');
                if (app) {{ app.style.transition='background 0.9s ease'; app.style.background='{bg}'; }}
                const body = window.parent.document.body;
                if (body) {{ body.style.transition='background 0.9s ease'; body.style.background='{bg}'; }}
            }})();
        </script>
    """, height=0)


# ─────────────────────────────────────────────
# NEW FEATURE FUNCTIONS
# ─────────────────────────────────────────────

def send_sms_alert(confidence_percent: int, location_name: str = "Unknown Location") -> bool:
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        body = (
            f"🚨 GeoSentinel ALERT\n"
            f"Landslide detected at {location_name}.\n"
            f"Confidence: {confidence_percent}%\n"
            f"Risk Level: HIGH\n"
            f"Immediate evacuation required.\n"
            f"Emergency: 108 | NDMA: 1078"
        )
        msg = client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=TWILIO_TO_NUMBER)
        return msg.sid is not None
    except Exception as e:
        st.error(f"SMS failed: {e}")
        return False


def send_email_alert(confidence_percent: int, location_name: str = "Unknown", annotated_path: str = None) -> bool:
    html_body = f"""
    <html><body style="background:#0a0e1a;color:#e8eaf0;font-family:sans-serif;padding:2rem;">
    <div style="max-width:600px;margin:auto;background:#111827;border-radius:16px;overflow:hidden;border:1px solid #2a3a5c;">
        <div style="background:linear-gradient(135deg,#ff5a3c,#ff9a3c);padding:1.5rem 2rem;">
            <h1 style="margin:0;font-size:1.4rem;color:#fff;">🚨 GeoSentinel — Landslide Detected</h1>
        </div>
        <div style="padding:2rem;">
            <p style="color:#f87171;">Landslide at <b>{location_name}</b> — Confidence: <b>{confidence_percent}%</b></p>
            <table style="width:100%;border-collapse:collapse;margin:1.5rem 0;">
                <tr style="background:#1e2d45;"><td style="padding:0.6rem 1rem;color:#7a8399;">Confidence</td><td style="padding:0.6rem 1rem;color:#ff5a3c;font-weight:bold;">{confidence_percent}%</td></tr>
                <tr><td style="padding:0.6rem 1rem;color:#7a8399;">Risk Level</td><td style="padding:0.6rem 1rem;color:#ef4444;font-weight:bold;">HIGH</td></tr>
                <tr style="background:#1e2d45;"><td style="padding:0.6rem 1rem;color:#7a8399;">Location</td><td style="padding:0.6rem 1rem;color:#e8eaf0;">{location_name}</td></tr>
            </table>
            <div style="background:#1a0808;border-left:4px solid #ef4444;padding:1rem 1.5rem;border-radius:8px;">
                <b style="color:#fca5a5;">Actions Required:</b>
                <ul style="color:#f87171;margin:0.5rem 0;padding-left:1.2rem;">
                    <li>Evacuate slopes and valleys immediately</li>
                    <li>Alert local disaster management authorities</li>
                    <li>NDMA: 1078 | Helpline: 108 | Police: 100</li>
                </ul>
            </div>
            <p style="color:#3a4a60;font-size:0.75rem;margin-top:2rem;">Auto-generated by GeoSentinel AI. Annotated image attached.</p>
        </div>
    </div></body></html>
    """
    message = Mail(
        from_email=ALERT_FROM_EMAIL,
        to_emails=ALERT_TO_EMAIL,
        subject=f"🚨 GeoSentinel ALERT — Landslide ({confidence_percent}% confidence)",
        html_content=html_body
    )
    if annotated_path and os.path.exists(annotated_path):
        with open(annotated_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        message.attachment = Attachment(
            FileContent(encoded), FileName("prediction.jpg"),
            FileType("image/jpeg"), Disposition("attachment")
        )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        r = sg.send(message)
        return r.status_code in (200, 202)
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False


def analyze_video(video_path: str, conf_threshold: float = 0.5, frame_skip: int = 5) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = "annotated_video.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_idx = analyzed = detected_frames = 0
    max_conf = 0.0
    frame_results = []
    pb = st.progress(0, text="Analysing video frames…")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_path = tmp.name
            tmp.close()  # ✅ VERY IMPORTANT (fixes your error)
            cv2.imwrite(tmp_path, frame)
            res = model.predict(tmp_path, conf=conf_threshold, verbose=False)
            try:
                os.unlink(tmp_path)  # ✅ safe delete
            except PermissionError:
                pass
            detected   = len(res[0].boxes) > 0
            confidence = float(res[0].boxes.conf.max()) if detected else 0.0
            if detected:
                detected_frames += 1
                max_conf = max(max_conf, confidence)
                annotated = res[0].plot()
            else:
                annotated = frame
            frame_results.append({"frame": frame_idx, "time_sec": round(frame_idx / fps, 2),
                                   "confidence": round(confidence * 100, 1), "detected": detected})
            analyzed += 1
        else:
            annotated = frame
        out.write(annotated)
        pb.progress(min(int(frame_idx / max(total, 1) * 100), 100), text=f"Frame {frame_idx}/{total}")
        frame_idx += 1
    cap.release()
    out.release()
    pb.empty()
    return {"total_frames": total, "analyzed_frames": analyzed,
            "detected_frames": detected_frames, "max_confidence": round(max_conf * 100, 1),
            "output_path": output_path, "frame_results": frame_results}
#heatmap
def show_heatmap():
    try:
        from database import FILE_NAME
        df = pd.read_csv(FILE_NAME, on_bad_lines='skip')
        if df.empty:
            st.warning("No data available yet.")
            return

        # Create map centered on average location
        m = folium.Map(
            location=[df["latitude"].mean(), df["longitude"].mean()],
            zoom_start=7,
            tiles="CartoDB dark_matter"
        )

        # Heatmap data
        heat_data = [
    [row["latitude"], row["longitude"], row["confidence"]/100]
    for _, row in df.iterrows()]

        HeatMap(heat_data).add_to(m)

        st.markdown("### 🔥 Landslide Heatmap")
        st_folium(m, width=None, height=500)

    except FileNotFoundError:
        st.warning("No dataset found yet. Run detection first.")

def render_map(detected: bool, confidence_percent: int = 0) -> str:
    st.markdown("""
    <div style='font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;color:#7a8399;margin:2rem 0 1rem;'>
        🗺️ Geolocation & Risk Map
    </div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        lat = st.number_input("Latitude", value=12.4244, format="%.4f", key="lat_input")
    with c2:
        lon = st.number_input("Longitude", value=75.7382, format="%.4f", key="lon_input")
    with c3:
        place = st.text_input("Location Name", value="Kodagu, Karnataka")
    location_name = f"{place} ({lat:.4f}°N, {lon:.4f}°E)"
    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB dark_matter",
                   attr="© OpenStreetMap | © CartoDB")
    if detected:
        folium.CircleMarker([lat, lon], radius=30, color="#ef4444", fill=True,
                            fill_color="#ef4444", fill_opacity=0.15, weight=2,
                            popup=folium.Popup(f"<b>⚠️ Landslide Detected</b><br>Conf: {confidence_percent}%<br>{place}", max_width=200)).add_to(m)
        folium.Marker([lat, lon], tooltip=f"⚠️ {confidence_percent}%",
                      popup=folium.Popup(f"<b>🚨 DANGER ZONE</b><br>{place}", max_width=200),
                      icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon")).add_to(m)
        folium.Circle([lat, lon], radius=1000, color="#ff5a3c", fill=False,
                      weight=1.5, dash_array="6", tooltip="1 km evacuation radius").add_to(m)
    else:
        folium.Marker([lat, lon], tooltip="✅ Safe",
                      popup=folium.Popup(f"<b>✅ Area is Safe</b><br>{place}", max_width=200),
                      icon=folium.Icon(color="green", icon="ok-sign", prefix="glyphicon")).add_to(m)
    st_folium(m, width=None, height=420, returned_objects=[])
    st.markdown(f"""
    <div style='font-size:0.78rem;color:#7a8399;margin-top:0.5rem;'>
        📍 Pinned: <span style='color:#e8eaf0;'>{location_name}</span>
        {"&nbsp;·&nbsp;<span style='color:#ef4444;'>🔴 1 km evacuation radius shown</span>" if detected else ""}
    </div>""", unsafe_allow_html=True)
    return location_name, lat, lon


# ─────────────────────────────────────────────
# GLOBAL STYLES (unchanged from original)
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,.stApp{background:#0a0e1a!important;color:#e8eaf0!important;font-family:'DM Sans',sans-serif!important;transition:background 0.9s ease!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 3rem 4rem!important;max-width:1280px!important;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:#0a0e1a;}::-webkit-scrollbar-thumb{background:#2a3a5c;border-radius:2px;}
.hero{text-align:center;padding:3.5rem 0 2rem;position:relative;}
.hero::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(255,80,60,0.12) 0%,transparent 70%);pointer-events:none;}
.hero-eyebrow{font-family:'DM Sans',sans-serif;font-size:0.7rem;font-weight:500;letter-spacing:0.25em;text-transform:uppercase;color:#ff5a3c;margin-bottom:0.75rem;}
.hero-title{font-family:'Syne',sans-serif;font-size:clamp(2.4rem,5vw,4rem);font-weight:800;line-height:1.1;letter-spacing:-0.02em;color:#ffffff;}
.hero-title span{background:linear-gradient(135deg,#ff5a3c 0%,#ff9a3c 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{font-size:1rem;color:#7a8399;margin-top:0.75rem;font-weight:300;}
.hero-line{width:48px;height:3px;background:linear-gradient(90deg,#ff5a3c,#ff9a3c);margin:1.5rem auto 0;border-radius:2px;}
.img-card-label{padding:0.6rem 1rem;font-size:0.7rem;font-weight:500;letter-spacing:0.15em;text-transform:uppercase;color:#7a8399;border-bottom:1px solid #1e2d45;display:flex;align-items:center;gap:0.5rem;}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block;}.dot-orange{background:#ff5a3c;}.dot-green{background:#22c55e;}
.slim-divider{height:1px;background:linear-gradient(90deg,transparent,#2a3a5c 30%,#2a3a5c 70%,transparent);margin:2rem 0;}
.alert-danger{background:linear-gradient(135deg,#1a0808,#2d0f0f);border:1px solid #7f1d1d;border-left:4px solid #ef4444;border-radius:14px;padding:2rem 2.5rem;margin:2rem 0;position:relative;overflow:hidden;animation:pulseRed 2.5s ease-in-out infinite;}
.alert-danger::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 50% 0%,rgba(239,68,68,0.15) 0%,transparent 60%);}
@keyframes pulseRed{0%,100%{box-shadow:0 0 0 rgba(239,68,68,0);}50%{box-shadow:0 0 40px rgba(239,68,68,0.25);}}
.alert-danger-icon{font-size:2.5rem;margin-bottom:0.5rem;}
.alert-danger-title{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#fca5a5;letter-spacing:0.04em;}
.alert-danger-sub{color:#f87171;font-size:0.9rem;margin-top:0.3rem;font-weight:300;}
.alert-safe{background:linear-gradient(135deg,#051a0e,#0a2418);border:1px solid #14532d;border-left:4px solid #22c55e;border-radius:14px;padding:2rem 2.5rem;margin:2rem 0;}
.alert-safe-title{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#86efac;letter-spacing:0.04em;}
.alert-safe-sub{color:#4ade80;font-size:0.9rem;margin-top:0.3rem;font-weight:300;}
.stats-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1.5rem 0;}
.stat-card{background:#111827;border:1px solid #1e2d45;border-radius:12px;padding:1.25rem 1.5rem;}
.stat-value{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#ff5a3c;line-height:1;}
.stat-label{font-size:0.7rem;font-weight:500;letter-spacing:0.12em;text-transform:uppercase;color:#7a8399;margin-top:0.4rem;}
.conf-bar-wrap{margin:1.5rem 0;}.conf-bar-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;color:#7a8399;}
.conf-bar-track{background:#1e2d45;border-radius:4px;height:6px;overflow:hidden;}.conf-bar-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,#ff5a3c,#ff9a3c);transition:width 1s cubic-bezier(.4,0,.2,1);}
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1.5rem 0;}
.info-card{background:#111827;border:1px solid #1e2d45;border-radius:12px;padding:1.25rem 1.5rem;}
.info-card-title{font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;color:#e8eaf0;margin-bottom:0.75rem;display:flex;align-items:center;gap:0.5rem;}
.info-card ul{list-style:none;padding:0;}.info-card li{font-size:0.82rem;color:#7a8399;padding:0.3rem 0;border-bottom:1px solid #1a2640;display:flex;align-items:center;gap:0.5rem;}.info-card li:last-child{border-bottom:none;}.info-card li::before{content:'→';color:#ff5a3c;font-size:0.7rem;flex-shrink:0;}
section[data-testid="stSidebar"]{background:#0d1221!important;border-right:1px solid #1e2d45!important;}
section[data-testid="stSidebar"] *{color:#e8eaf0!important;}
[data-testid="stFileUploadDropzone"]{background:#111827!important;border:1.5px dashed #2a3a5c!important;border-radius:14px!important;padding:2rem!important;transition:border-color 0.3s!important;}
[data-testid="stFileUploadDropzone"]:hover{border-color:#ff5a3c!important;}
[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#ff5a3c,#ff9a3c)!important;border-radius:4px!important;}
[data-testid="stProgress"]>div{background:#1e2d45!important;border-radius:4px!important;height:6px!important;}
[data-testid="stDownloadButton"] button{background:linear-gradient(135deg,#ff5a3c,#ff9a3c)!important;color:white!important;border:none!important;border-radius:10px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;letter-spacing:0.05em!important;padding:0.6rem 1.5rem!important;transition:opacity 0.2s!important;box-shadow:0 4px 20px rgba(255,90,60,0.3)!important;}
[data-testid="stImage"] img{border-radius:10px!important;}
[data-testid="stHorizontalBlock"]{gap:1.5rem!important;}

/* Alert buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e2d45, #2a3a5c) !important;
    color: #e8eaf0 !important;
    border: 1px solid #2a3a5c !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: #ff5a3c !important;
    color: #ff5a3c !important;
}

iframe { border-radius: 14px !important; border: 1px solid #1e2d45 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#fff;letter-spacing:-0.01em;'>🛰 GeoSentinel</div>
        <div style='font-size:0.7rem;color:#7a8399;letter-spacing:0.15em;text-transform:uppercase;margin-top:0.2rem;'>Landslide Detection AI</div>
    </div>
    <div style='height:1px;background:#1e2d45;margin:0.5rem 0 1.5rem;'></div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:#7a8399;margin-bottom:0.75rem;'>How It Works</div>", unsafe_allow_html=True)
    for num, desc in [("01","Upload image or drone video"),("02","AI model segments terrain"),("03","Confidence score computed"),("04","Alert issued + map pinned"),("05","SMS & Email sent to officers")]:
        st.markdown(f"""
        <div style='display:flex;gap:0.75rem;align-items:flex-start;margin-bottom:1rem;'>
            <div style='font-family:Syne,sans-serif;font-weight:800;font-size:0.7rem;color:#ff5a3c;padding-top:2px;flex-shrink:0;'>{num}</div>
            <div style='font-size:0.82rem;color:#7a8399;line-height:1.5;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='height:1px;background:#1e2d45;margin:1rem 0;'></div>
    <div style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:#7a8399;margin-bottom:0.75rem;'>Model Info</div>
    """, unsafe_allow_html=True)
    for label, val in [("Architecture","YOLOv8-Seg"),("Input","640×640 px"),("Threshold","0.50 conf"),("Classes","Landslide / Safe"),("Alerts","SMS + Email"),("Map","Folium / OSM")]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:0.4rem 0;border-bottom:1px solid #1e2d45;'>
            <span style='font-size:0.78rem;color:#7a8399;'>{label}</span>
            <span style='font-size:0.78rem;color:#e8eaf0;font-weight:500;'>{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;color:#3a4a60;text-align:center;line-height:1.6;'>Built for rapid disaster response.<br>For emergency use only.</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Terrain Analysis</div>
    <div class="hero-title">Detect <span>Landslides</span><br>Before Disaster Strikes</div>
    <div class="hero-sub">Image analysis · Drone video scanning · Live map · SMS & Email alerts</div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷  Image Analysis", "🎞️  Drone / Video Analysis"])
# ══════════════════════════════════════════════
# TAB 1 — IMAGE ANALYSIS
# ══════════════════════════════════════════════
with tab1:
    uploaded_file = st.file_uploader(
        "Drop your image here or click to browse",
        type=["jpg", "png", "jpeg"],
        label_visibility="visible",
        key="img_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown("<div class='slim-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;color:#7a8399;margin-bottom:1rem;'>Analysis Output</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='img-card-label'><span class='dot dot-orange'></span> Original Image</div>", unsafe_allow_html=True)
            st.image(image, width="stretch")

        image.save("temp.jpg")
        results = model.predict("temp.jpg", conf=0.5)

        # ── DETECTIONS FOUND ──
        if len(results[0].boxes) > 0:
            conf    = float(results[0].boxes.conf.max())
            percent = int(conf * 100)
            results[0].save(filename="output.jpg")

            with col2:
                st.markdown("<div class='img-card-label'><span class='dot dot-orange'></span> AI Prediction</div>", unsafe_allow_html=True)
                st.image("output.jpg", width="stretch")

            st.markdown("<div class='slim-divider'></div>", unsafe_allow_html=True)

            # HIGH CONFIDENCE → DANGER
            if conf >= 0.5:
                set_background("danger")

                st.markdown(f"""
                <div class="alert-danger">
                    <div class="alert-danger-icon">🚨</div>
                    <div class="alert-danger-title">LANDSLIDE DETECTED</div>
                    <div class="alert-danger-sub">Immediate action required — evacuate affected zones</div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="stats-grid">
                    <div class="stat-card"><div class="stat-value">{percent}%</div><div class="stat-label">Confidence</div></div>
                    <div class="stat-card"><div class="stat-value" style="color:#ef4444;">HIGH</div><div class="stat-label">Risk Level</div></div>
                    <div class="stat-card"><div class="stat-value">{len(results[0].boxes)}</div><div class="stat-label">Detections</div></div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="conf-bar-wrap">
                    <div class="conf-bar-header"><span>Confidence Score</span><span style="color:#ff5a3c;font-weight:600;">{percent}%</span></div>
                    <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{percent}%"></div></div>
                </div>""", unsafe_allow_html=True)

                # ── 🗺️ MAP ───────────────────────────────────────────
                location_name, lat, lon = render_map(detected=True,confidence_percent=percent)
                # ✅ SAVE DATA
    
                if st.button("💾 Save to Database", key="save_img"):
                    save_landslide(lat, lon, percent, location_name)
                if percent >= 70:
                    if st.button("🚨 Send Auto Alerts", key="auto_alert_img"):
                        send_sms_alert(percent, location_name)
                        send_email_alert(percent, location_name, "output.jpg")

                # ── 📡 ALERT BUTTONS ─────────────────────────────────
                st.markdown("""
                <div style='font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;
                            color:#7a8399;margin:1.5rem 0 0.75rem;'>📡 Send Alerts</div>
                """, unsafe_allow_html=True)

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("📱 Send SMS Alert", key="sms_img"):
                        with st.spinner("Sending SMS…"):
                            ok = send_sms_alert(percent, location_name)
                        st.success("✅ SMS sent!") if ok else st.error("SMS failed.")
                with btn_col2:
                    if st.button("📧 Send Email Alert", key="email_img"):
                        with st.spinner("Sending email…"):
                            ok = send_email_alert(percent, location_name, "output.jpg")
                        st.success("✅ Email sent with annotated image!") if ok else st.error("Email failed.")

                # GIF + Audio (original)
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcHJmenc0dG11ZXlmcTJ2anAzcTdsdjNiczdvcWZucGpwZHEyd2VvYyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/51Uiuy5QBZNkoF3b2Z/giphy.gif", width=280)

                alert_text = f"Warning! Landslide detected with {percent} percent confidence. Risk level is high. Please evacuate immediately. Emergency Numbers. Disaster Helpline number — 108. Fire Brigade number — 101"
                speak_alert(alert_text)
                st.audio("alert.mp3")

                st.markdown("""
                <div class="info-grid">
                    <div class="info-card"><div class="info-card-title">🚑 Emergency Contacts</div><ul><li>Disaster Helpline — 108</li><li>Police — 100</li><li>Fire Brigade — 101</li><li>NDMA India — 1078</li></ul></div>
                    <div class="info-card"><div class="info-card-title">🛡️ Safety Guidelines</div><ul><li>Evacuate slopes immediately</li><li>Avoid riverbanks & valleys</li><li>Alert local authorities</li><li>Monitor rainfall alerts</li></ul></div>
                </div>""", unsafe_allow_html=True)

                with open("output.jpg", "rb") as f:
                    st.download_button("📥 Download Annotated Result", data=f,
                                       file_name="geosentinel_prediction.jpg", mime="image/jpeg")

            # LOW CONFIDENCE → SAFE
            else:
                set_background("safe")
                location_name = render_map(detected=False)

                st.markdown(f"""
                <div class="alert-safe">
                    <div style="font-size:2rem;margin-bottom:0.5rem;">✅</div>
                    <div class="alert-safe-title">NO SIGNIFICANT THREAT</div>
                    <div class="alert-safe-sub">Low-confidence detection — terrain appears stable</div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="stats-grid">
                    <div class="stat-card"><div class="stat-value" style="color:#22c55e;">{percent}%</div><div class="stat-label">Confidence</div></div>
                    <div class="stat-card"><div class="stat-value" style="color:#22c55e;">LOW</div><div class="stat-label">Risk Level</div></div>
                    <div class="stat-card"><div class="stat-value">{len(results[0].boxes)}</div><div class="stat-label">Detections</div></div>
                </div>""", unsafe_allow_html=True)

                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image("https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3NXF4amx2cjJkMm0xMXpzc2RjcXV0M3draGJudG5yZmNybWVieG12ayZlcD12MV9naWZzX3NlYXJjaCZjdD1n/qqJiWx3tggnExIN2re/giphy.gif", width=280)

        # NO DETECTIONS
        else:
            set_background("safe")
            location_name = render_map(detected=False)

            with col2:
                st.markdown("""
                <div style='height:100%;display:flex;align-items:center;justify-content:center;
                            background:#0d1a2a;border-radius:12px;padding:3rem 1rem;text-align:center;'>
                    <div>
                        <div style='font-size:3rem;margin-bottom:1rem;'>🌿</div>
                        <div style='font-family:Syne,sans-serif;font-weight:700;color:#86efac;font-size:1rem;'>No anomaly detected</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div class='slim-divider'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="alert-safe">
                <div style="font-size:2rem;margin-bottom:0.5rem;">✅</div>
                <div class="alert-safe-title">AREA IS SAFE</div>
                <div class="alert-safe-sub">No landslide features detected in this image</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="stats-grid">
                <div class="stat-card"><div class="stat-value" style="color:#22c55e;">0%</div><div class="stat-label">Risk Score</div></div>
                <div class="stat-card"><div class="stat-value" style="color:#22c55e;">NONE</div><div class="stat-label">Risk Level</div></div>
                <div class="stat-card"><div class="stat-value">0</div><div class="stat-label">Detections</div></div>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbnZobDFoczVmYmczdjF1MTU5NXIwMGdlcWJveGQycGM3aXMzcGNqNyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Fu3OjBQiCs3s0ZuLY3/giphy.gif", width=280)

    else:
        set_background("none")
        st.markdown("""
        <div style='text-align:center;padding:4rem 2rem;background:linear-gradient(145deg,#0d1221,#111827);
                    border:1px dashed #2a3a5c;border-radius:20px;margin:1.5rem 0;'>
            <div style='font-size:3.5rem;margin-bottom:1.25rem;opacity:0.8;'>🛰️</div>
            <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:#e8eaf0;margin-bottom:0.5rem;'>Awaiting Image Input</div>
            <div style='font-size:0.85rem;color:#3a4a60;max-width:360px;margin:0 auto;line-height:1.7;'>Upload a satellite image, drone capture, or field photo above to begin terrain analysis.</div>
            <div style='display:inline-flex;gap:2.5rem;margin-top:2.5rem;padding:1rem 2.5rem;background:#0a0e1a;border-radius:12px;border:1px solid #1e2d45;'>
                <div style='text-align:center;'><div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#ff5a3c;'>YOLOv8</div><div style='font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a60;margin-top:2px;'>Model</div></div>
                <div style='width:1px;background:#1e2d45;'></div>
                <div style='text-align:center;'><div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#ff5a3c;'>Seg</div><div style='font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a60;margin-top:2px;'>Type</div></div>
                <div style='width:1px;background:#1e2d45;'></div>
                <div style='text-align:center;'><div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#ff5a3c;'>0.50</div><div style='font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a60;margin-top:2px;'>Conf Threshold</div></div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — VIDEO / DRONE ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style='font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;color:#7a8399;margin:1rem 0 1.5rem;'>
        Upload drone footage, timelapse, or GIF for frame-by-frame analysis
    </div>""", unsafe_allow_html=True)

    video_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "gif"], key="vid_up")

    vc1, vc2 = st.columns(2)
    with vc1:
        frame_skip = st.slider("Analyze every N frames", 1, 30, 5, help="Higher = faster processing")
    with vc2:
        vid_conf = st.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05, key="vid_conf_slider")

    if video_file and st.button("🚀 Run Video Analysis", key="run_video"):
        tmp = tempfile.NamedTemporaryFile(suffix=f".{video_file.name.split('.')[-1]}", delete=False)
        tmp.write(video_file.read())
        tmp_path = tmp.name
        tmp.close()  # ✅ FIXS
        res = analyze_video(tmp_path, conf_threshold=vid_conf, frame_skip=frame_skip)
        try:
            os.unlink(tmp_path)
        except PermissionError:
            pass

        if "error" in res:
            st.error(res["error"])
        else:
            danger = res["detected_frames"] > 0
            color  = "#ef4444" if danger else "#22c55e"

            st.markdown(f"""
            <div style='background:{"#1a0808" if danger else "#051a0e"};
                        border:1px solid {"#7f1d1d" if danger else "#14532d"};
                        border-left:4px solid {color};border-radius:14px;
                        padding:1.5rem 2rem;margin:1rem 0;'>
                <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                            color:{"#fca5a5" if danger else "#86efac"};'>
                    {"🚨 LANDSLIDE FRAMES DETECTED" if danger else "✅ NO LANDSLIDE DETECTED"}
                </div>
                <div style='font-size:0.85rem;color:{"#f87171" if danger else "#4ade80"};margin-top:0.3rem;'>
                    {"Immediate review and alert recommended." if danger else "Video analysis complete — terrain appears stable."}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stats-grid">
                <div class="stat-card"><div class="stat-value">{res['total_frames']}</div><div class="stat-label">Total Frames</div></div>
                <div class="stat-card"><div class="stat-value" style="color:{color};">{res['detected_frames']}</div><div class="stat-label">Alert Frames</div></div>
                <div class="stat-card"><div class="stat-value" style="color:{color};">{res['max_confidence']}%</div><div class="stat-label">Peak Confidence</div></div>
            </div>""", unsafe_allow_html=True)

            # Frame Timeline
            if res["frame_results"]:
                st.markdown("<div style='font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;color:#7a8399;margin:1.5rem 0 0.5rem;'>Frame Timeline</div>", unsafe_allow_html=True)
                tl = "<div style='display:flex;flex-wrap:wrap;gap:4px;margin:0.5rem 0;'>"
                for fr in res["frame_results"]:
                    bg  = "#ef4444" if fr["detected"] else "#1e2d45"
                    tip = f"t={fr['time_sec']}s — {fr['confidence']}%"
                    tl += f"<div title='{tip}' style='width:12px;height:28px;background:{bg};border-radius:3px;'></div>"
                tl += "</div><div style='font-size:0.72rem;color:#3a4a60;'>🔴 Red = landslide detected in frame &nbsp; | &nbsp; Hover for timestamp & confidence</div>"
                st.markdown(tl, unsafe_allow_html=True)

            # Map for video
            st.markdown("<div class='slim-divider'></div>", unsafe_allow_html=True)
            vid_location, lat, lon = render_map(
    detected=danger,
    confidence_percent=res["max_confidence"]
    
)  
         
# ✅ STORE IN DATA CSV
            if res["max_confidence"] >= 50:
                save_landslide(lat, lon, res["max_confidence"], vid_location)
                if res["max_confidence"] >= 70:
                    pass
            # Alerts for video
            if danger and res["max_confidence"] > 70:
                st.markdown("<div style='font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;color:#7a8399;margin:1.5rem 0 0.75rem;'>📡 Send Alerts</div>", unsafe_allow_html=True)
                va1, va2 = st.columns(2)
                with va1:
                    if st.button("📱 Send SMS Alert", key="sms_vid"):
                        with st.spinner("Sending…"):
                            ok = send_sms_alert(res["max_confidence"], vid_location)
                        st.success("✅ SMS sent!") if ok else st.error("SMS failed.")
                with va2:
                    if st.button("📧 Send Email Alert", key="email_vid"):
                        with st.spinner("Sending…"):
                            ok = send_email_alert(res["max_confidence"], vid_location)
                        st.success("✅ Email sent!") if ok else st.error("Email failed.")

            # Download annotated video
            if os.path.exists(res["output_path"]):
                with open(res["output_path"], "rb") as vf:
                    st.download_button("📥 Download Annotated Video", data=vf,
                                       file_name="geosentinel_annotated.mp4", mime="video/mp4")

    elif not video_file:
        st.markdown("""
        <div style='text-align:center;padding:3rem 2rem;background:linear-gradient(145deg,#0d1221,#111827);
                    border:1px dashed #2a3a5c;border-radius:20px;margin:1rem 0;'>
            <div style='font-size:3rem;margin-bottom:1rem;opacity:0.7;'>🎞️</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#e8eaf0;'>Upload Drone Footage</div>
            <div style='font-size:0.82rem;color:#3a4a60;max-width:340px;margin:0.5rem auto;line-height:1.7;'>
                Supports MP4, AVI, MOV, GIF. Each frame is individually analysed by the YOLO model.
            </div>
        </div>""", unsafe_allow_html=True)
        # 🔥 Show Heatmap Button
# Initialize state
if "show_map" not in st.session_state:
    st.session_state.show_map = False

# Button
if st.button("🔥 Show Landslide Heatmap"):
    st.session_state.show_map = True

# Display map persistently
if st.session_state.show_map:
    show_heatmap()

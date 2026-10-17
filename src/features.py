# ═══════════════════════════════════════════════════════════════════
#  GeoSentinel — New Feature Modules
#  Add these imports and functions to your existing app.py
# ═══════════════════════════════════════════════════════════════════
#
#  FEATURES ADDED:
#   1. 📧 Email Alert via SendGrid
#   2. 📱 SMS Alert via Twilio
#   3. 🎞️ Video/GIF Frame-by-Frame Drone Footage Analysis
#   4. 🗺️ Map Integration (Folium) with GPS Pinning
#
# ═══════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────
# NEW IMPORTS — Add these to your existing imports
# ───────────────────────────────────────────
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import streamlit.components.v1 as components
import pyttsx3

# --- NEW ---
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
from io import BytesIO
import time


# ───────────────────────────────────────────
# CONFIGURATION — Add to your CONFIG section
# ───────────────────────────────────────────

# ── Twilio ──────────────────────────────────
TWILIO_ACCOUNT_SID = "YOUR_TWILIO_ACCOUNT_SID"     # from twilio.com/console
TWILIO_AUTH_TOKEN  = "YOUR_TWILIO_AUTH_TOKEN"       # from twilio.com/console
TWILIO_FROM_NUMBER = "+1XXXXXXXXXX"                  # your Twilio number
TWILIO_TO_NUMBER   = "+91XXXXXXXXXX"                 # recipient (field officer)

# ── SendGrid ─────────────────────────────────
SENDGRID_API_KEY   = "YOUR_SENDGRID_API_KEY"        # from app.sendgrid.com
ALERT_FROM_EMAIL   = "alerts@geosentinel.io"        # verified sender
ALERT_TO_EMAIL     = "officer@ndma.gov.in"          # recipient


# ═══════════════════════════════════════════════════════════════════
# FEATURE 1 — 📱 SMS ALERT via Twilio
# ═══════════════════════════════════════════════════════════════════

def send_sms_alert(confidence_percent: int, location_name: str = "Unknown Location") -> bool:
    """
    Sends an SMS alert to field officers when a landslide is detected.

    Args:
        confidence_percent : Detection confidence (0–100)
        location_name      : Human-readable location string from GPS input

    Returns:
        True if sent successfully, False otherwise.
    """
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message_body = (
            f"🚨 GeoSentinel ALERT\n"
            f"Landslide detected at {location_name}.\n"
            f"Confidence: {confidence_percent}%\n"
            f"Risk Level: HIGH\n"
            f"Immediate evacuation required.\n"
            f"Emergency: 108 | NDMA: 1078"
        )
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_FROM_NUMBER,
            to=TWILIO_TO_NUMBER
        )
        return message.sid is not None

    except Exception as e:
        st.error(f"SMS failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# FEATURE 2 — 📧 EMAIL ALERT via SendGrid
# ═══════════════════════════════════════════════════════════════════

def send_email_alert(
    confidence_percent: int,
    location_name: str = "Unknown Location",
    annotated_image_path: str = None
) -> bool:
    """
    Sends a rich HTML email alert with the annotated prediction image attached.

    Args:
        confidence_percent    : Detection confidence (0–100)
        location_name         : Human-readable location string
        annotated_image_path  : Path to output.jpg (annotated result)

    Returns:
        True if sent successfully, False otherwise.
    """
    html_body = f"""
    <html>
    <body style="background:#0a0e1a; color:#e8eaf0; font-family:sans-serif; padding:2rem;">
        <div style="max-width:600px; margin:auto; background:#111827;
                    border-radius:16px; overflow:hidden; border:1px solid #2a3a5c;">
            <div style="background:linear-gradient(135deg,#ff5a3c,#ff9a3c);
                        padding:1.5rem 2rem;">
                <h1 style="margin:0; font-size:1.4rem; color:#fff;">
                    🚨 GeoSentinel — Landslide Detected
                </h1>
            </div>
            <div style="padding:2rem;">
                <p style="color:#f87171; font-size:1rem;">
                    A landslide has been detected with <strong>{confidence_percent}%</strong>
                    confidence at <strong>{location_name}</strong>.
                </p>
                <table style="width:100%; border-collapse:collapse; margin:1.5rem 0;">
                    <tr style="background:#1e2d45;">
                        <td style="padding:0.6rem 1rem; color:#7a8399; font-size:0.85rem;">Confidence</td>
                        <td style="padding:0.6rem 1rem; color:#ff5a3c; font-weight:bold;">{confidence_percent}%</td>
                    </tr>
                    <tr>
                        <td style="padding:0.6rem 1rem; color:#7a8399; font-size:0.85rem;">Risk Level</td>
                        <td style="padding:0.6rem 1rem; color:#ef4444; font-weight:bold;">HIGH</td>
                    </tr>
                    <tr style="background:#1e2d45;">
                        <td style="padding:0.6rem 1rem; color:#7a8399; font-size:0.85rem;">Location</td>
                        <td style="padding:0.6rem 1rem; color:#e8eaf0;">{location_name}</td>
                    </tr>
                </table>
                <div style="background:#1a0808; border-left:4px solid #ef4444;
                            padding:1rem 1.5rem; border-radius:8px; margin:1rem 0;">
                    <strong style="color:#fca5a5;">Recommended Actions:</strong>
                    <ul style="color:#f87171; margin:0.5rem 0; padding-left:1.2rem;">
                        <li>Evacuate slopes and valleys immediately</li>
                        <li>Alert local disaster management authorities</li>
                        <li>Call NDMA: 1078 | Disaster Helpline: 108</li>
                    </ul>
                </div>
                <p style="color:#3a4a60; font-size:0.75rem; margin-top:2rem;">
                    This alert was generated automatically by GeoSentinel AI.
                    Annotated image is attached.
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    message = Mail(
        from_email=ALERT_FROM_EMAIL,
        to_emails=ALERT_TO_EMAIL,
        subject=f"🚨 GeoSentinel ALERT — Landslide Detected ({confidence_percent}% confidence)",
        html_content=html_body
    )

    # Attach annotated image if available
    if annotated_image_path and os.path.exists(annotated_image_path):
        with open(annotated_image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        attachment = Attachment(
            FileContent(encoded),
            FileName("geosentinel_prediction.jpg"),
            FileType("image/jpeg"),
            Disposition("attachment")
        )
        message.attachment = attachment

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        return response.status_code in (200, 202)
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# FEATURE 3 — 🎞️ VIDEO / GIF FRAME-BY-FRAME ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_video(video_path: str, model: YOLO, conf_threshold: float = 0.5,
                  frame_skip: int = 5) -> dict:
    """
    Extracts frames from a video/GIF and runs YOLO inference on each.
    Saves an annotated output video and returns a summary.

    Args:
        video_path      : Path to the uploaded video file
        model           : Loaded YOLO model instance
        conf_threshold  : Confidence threshold for detections
        frame_skip      : Analyze every N-th frame (speeds up processing)

    Returns:
        dict with keys: total_frames, analyzed_frames, detected_frames,
                        max_confidence, output_path, frame_results
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = "annotated_video.mp4"
    fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer  = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx       = 0
    analyzed        = 0
    detected_frames = 0
    max_conf        = 0.0
    frame_results   = []   # [{frame, confidence, detected}, ...]

    progress_bar = st.progress(0, text="Analysing video frames…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # Save frame temporarily for YOLO
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tmp.name, frame)
            results = model.predict(tmp.name, conf=conf_threshold, verbose=False)
            os.unlink(tmp.name)

            detected   = len(results[0].boxes) > 0
            confidence = float(results[0].boxes.conf.max()) if detected else 0.0

            if detected:
                detected_frames += 1
                max_conf = max(max_conf, confidence)
                # Draw annotated frame from YOLO result
                annotated = results[0].plot()
            else:
                annotated = frame

            frame_results.append({
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 2),
                "confidence": round(confidence * 100, 1),
                "detected": detected
            })
            analyzed += 1
        else:
            annotated = frame

        out_writer.write(annotated)

        progress = min(int((frame_idx / max(total, 1)) * 100), 100)
        progress_bar.progress(progress, text=f"Frame {frame_idx}/{total}")
        frame_idx += 1

    cap.release()
    out_writer.release()
    progress_bar.empty()

    return {
        "total_frames"    : total,
        "analyzed_frames" : analyzed,
        "detected_frames" : detected_frames,
        "max_confidence"  : round(max_conf * 100, 1),
        "output_path"     : output_path,
        "frame_results"   : frame_results
    }


def render_video_analysis_ui(model: YOLO):
    """
    Renders the full video upload + analysis UI block.
    Call this inside your main Streamlit app.
    """
    st.markdown("""
    <div style='font-size:0.7rem; letter-spacing:0.18em; text-transform:uppercase;
                color:#7a8399; margin:2rem 0 1rem;'>
        🎞️ Drone / Video Analysis
    </div>
    """, unsafe_allow_html=True)

    video_file = st.file_uploader(
        "Upload drone footage or timelapse video",
        type=["mp4", "avi", "mov", "gif"],
        key="video_uploader"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        frame_skip = st.slider("Analyze every N frames", 1, 30, 5,
                               help="Higher = faster but less accurate")
    with col_b:
        vid_conf = st.slider("Detection confidence threshold", 0.1, 1.0, 0.5, 0.05,
                             key="vid_conf")

    if video_file and st.button("🚀 Run Video Analysis", key="run_video"):
        tmp_vid = tempfile.NamedTemporaryFile(suffix=f".{video_file.name.split('.')[-1]}", delete=False)
        tmp_vid.write(video_file.read())
        tmp_vid.flush()

        results = analyze_video(tmp_vid.name, model, conf_threshold=vid_conf, frame_skip=frame_skip)
        os.unlink(tmp_vid.name)

        if "error" in results:
            st.error(results["error"])
            return

        # ── Summary Stats ──
        danger = results["detected_frames"] > 0
        color  = "#ef4444" if danger else "#22c55e"
        label  = "LANDSLIDE FRAMES DETECTED" if danger else "NO LANDSLIDE DETECTED"

        st.markdown(f"""
        <div style='background:{"#1a0808" if danger else "#051a0e"};
                    border:1px solid {"#7f1d1d" if danger else "#14532d"};
                    border-left:4px solid {color};
                    border-radius:14px; padding:1.5rem 2rem; margin:1rem 0;'>
            <div style='font-family:Syne,sans-serif; font-size:1.3rem;
                        font-weight:800; color:{"#fca5a5" if danger else "#86efac"};'>
                {"🚨" if danger else "✅"} {label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{results['total_frames']}</div>
                <div class="stat-label">Total Frames</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color:{color};">{results['detected_frames']}</div>
                <div class="stat-label">Alert Frames</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color:{color};">{results['max_confidence']}%</div>
                <div class="stat-label">Peak Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Frame Timeline ──
        if results["frame_results"]:
            st.markdown("""
            <div style='font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase;
                        color:#7a8399; margin:1.5rem 0 0.5rem;'>Frame Timeline</div>
            """, unsafe_allow_html=True)

            timeline_html = "<div style='display:flex; flex-wrap:wrap; gap:4px; margin:0.5rem 0;'>"
            for fr in results["frame_results"]:
                bg = "#ef4444" if fr["detected"] else "#1e2d45"
                tip = f"t={fr['time_sec']}s — {fr['confidence']}%"
                timeline_html += f"""
                <div title='{tip}' style='width:12px; height:28px; background:{bg};
                     border-radius:3px; cursor:pointer;' ></div>"""
            timeline_html += "</div><div style='font-size:0.72rem; color:#3a4a60;'>"\
                             "🔴 Red = landslide detected in frame</div>"
            st.markdown(timeline_html, unsafe_allow_html=True)
        # 🗺️ SHOW MAP
        location_name, lat, lon = render_map_ui(
    detected=danger,
    confidence_percent=results["max_confidence"]
)
        # ── Download annotated video ──
        if os.path.exists(results["output_path"]):
            with open(results["output_path"], "rb") as vf:
                st.download_button(
                    "📥 Download Annotated Video",
                    data=vf,
                    file_name="geosentinel_annotated.mp4",
                    mime="video/mp4"
                )


# ═══════════════════════════════════════════════════════════════════
# FEATURE 4 — 🗺️ MAP INTEGRATION (Folium + GPS Pinning)
# ═══════════════════════════════════════════════════════════════════

def render_map_ui(
    detected: bool,
    confidence_percent: int = 0
) -> str:
    """
    Renders a Folium map with a risk pin at user-entered GPS coordinates.
    Also returns the location name string for use in SMS/email alerts.

    Args:
        detected           : Whether a landslide was detected
        confidence_percent : Confidence score (0–100)

    Returns:
        location_name string (e.g. "Kodagu, Karnataka (12.42°N, 75.73°E)")
    """
    st.markdown("""
    <div style='font-size:0.7rem; letter-spacing:0.18em; text-transform:uppercase;
                color:#7a8399; margin:2rem 0 1rem;'>
        🗺️ Geolocation & Risk Map
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        lat = st.number_input("Latitude", value=12.4244, format="%.4f",
                              help="e.g. 12.4244 for Kodagu, Karnataka")
    with col2:
        lon = st.number_input("Longitude", value=75.7382, format="%.4f",
                              help="e.g. 75.7382 for Kodagu, Karnataka")
    with col3:
        place_name = st.text_input("Location Name", value="Kodagu, Karnataka",
                                   help="Human-readable place name")

    location_name = f"{place_name} ({lat:.4f}°N, {lon:.4f}°E)"

    # ── Build Folium Map ──
    tile = "CartoDB dark_matter"
    m = folium.Map(
        location=[lat, lon],
        zoom_start=11,
        tiles=tile,
        attr="© OpenStreetMap | © CartoDB"
    )

    if detected:
        # Pulsing red circle for danger
        folium.CircleMarker(
            location=[lat, lon],
            radius=30,
            color="#ef4444",
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.15,
            weight=2,
            popup=folium.Popup(
                f"<b style='color:#ef4444'>⚠️ Landslide Detected</b><br>"
                f"Confidence: {confidence_percent}%<br>"
                f"{place_name}",
                max_width=200
            )
        ).add_to(m)

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                f"<b>🚨 DANGER ZONE</b><br>{place_name}<br>Confidence: {confidence_percent}%",
                max_width=200
            ),
            tooltip=f"⚠️ Landslide — {confidence_percent}%",
            icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon")
        ).add_to(m)

        # Risk radius ring (1 km)
        folium.Circle(
            location=[lat, lon],
            radius=1000,        # metres
            color="#ff5a3c",
            fill=False,
            weight=1.5,
            dash_array="6",
            tooltip="1 km evacuation radius"
        ).add_to(m)

    else:
        # Green safe marker
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                f"<b style='color:#22c55e'>✅ Area is Safe</b><br>{place_name}",
                max_width=200
            ),
            tooltip="✅ No landslide detected",
            icon=folium.Icon(color="green", icon="ok-sign", prefix="glyphicon")
        ).add_to(m)

    # Render map in Streamlit
    st_folium(m, width=None, height=420, returned_objects=[])

    st.markdown(f"""
    <div style='font-size:0.78rem; color:#7a8399; margin-top:0.5rem;'>
        📍 Pinned: <span style='color:#e8eaf0;'>{location_name}</span>
        {"&nbsp;·&nbsp;<span style='color:#ef4444;'>🔴 Risk radius: 1 km shown</span>" if detected else ""}
    </div>
    """, unsafe_allow_html=True)

    return location_name, lat, lon


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION GUIDE — How to wire everything into your existing app
# ═══════════════════════════════════════════════════════════════════
#
# In your existing ANALYSIS section, after a detection is confirmed,
# add the following code blocks in order:
#
# ── STEP 1: Map (always show after upload) ──────────────────────
#
#   location_name = render_map_ui(
#       detected=True,          # or False
#       confidence_percent=percent
#   )
#
# ── STEP 2: Alerts (only on HIGH confidence detection) ──────────
#
#   col_sms, col_email = st.columns(2)
#   with col_sms:
#       if st.button("📱 Send SMS Alert", key="sms_btn"):
#           ok = send_sms_alert(percent, location_name)
#           if ok:
#               st.success("✅ SMS sent successfully!")
#           else:
#               st.error("SMS failed — check Twilio config.")
#
#   with col_email:
#       if st.button("📧 Send Email Alert", key="email_btn"):
#           ok = send_email_alert(percent, location_name, "output.jpg")
#           if ok:
#               st.success("✅ Email sent with annotated image!")
#           else:
#               st.error("Email failed — check SendGrid config.")
#
# ── STEP 3: Video Analysis (separate tab or section) ────────────
#
#   render_video_analysis_ui(model)
#
# ── RECOMMENDED: Use tabs to organise ───────────────────────────
#
#   tab1, tab2 = st.tabs(["📷 Image Analysis", "🎞️ Video Analysis"])
#   with tab1:
#       # your existing image analysis code
#   with tab2:
#       render_video_analysis_ui(model)
#
# ═══════════════════════════════════════════════════════════════════
# INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════
#
#   pip install twilio sendgrid folium streamlit-folium opencv-python
#
# ═══════════════════════════════════════════════════════════════════
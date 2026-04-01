import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import sys
import os
import pandas as pd
import time

# Add local path so internal imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from detection.recognition.pipeline.pipeline import Pipeline
from database.rto_mock import get_vehicle_info, HARDCODED_RECORDS

st.set_page_config(
    page_title="Lovable RTO Intelligence ✨", 
    layout="wide", 
    page_icon="✨",
    initial_sidebar_state="expanded"
)

# Initialize pipeline once and cache it
@st.cache_resource
def load_pipeline():
    return Pipeline()

pipeline = load_pipeline()

# Inject the Hardcoded Mock Records so the Database isn't "Pristine"
if 'db_records' not in st.session_state:
    st.session_state.db_records = []
    for rec in HARDCODED_RECORDS.values():
        st.session_state.db_records.append({
            "License Plate": rec["plate"],
            "Target Country": "India",
            "State Region": rec["state"],
            "Owner": rec["owner"],
            "Type of Vehicle": rec["type_of_vehicle"],
            "Vehicle Model": rec["model"],
            "Registration Date": rec["registration_date"],
            "Expiration Date": rec["expiry"],
            "Status": rec["status"]
        })

# EXTREME LOVABLE-INSPIRED ULTRA-PREMIUM CSS
st.markdown("""
    <style>
/* ABSOLUTE BEAUTY: LOVABLE UI V4 (Mesh Gradient & Elegant Pastels) */
    .stApp { 
        background-color: #f8fafc;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,0) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,0) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,0) 0, transparent 50%);
        background-image: linear-gradient(140deg, #f0f4ff 0%, #fae8ff 50%, #f0fdf4 100%);
        color: #0f172a;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .metric-box {
        background: rgba(255, 255, 255, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 32px;
        padding: 40px;
        margin-bottom: 30px;
        box-shadow: 
            0 4px 6px -1px rgba(0, 0, 0, 0.02),
            0 20px 40px -10px rgba(99, 102, 241, 0.1), 
            inset 0 0 0 1px rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        transition: all 0.5s cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    .metric-box:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 10px 15px -3px rgba(0, 0, 0, 0.03),
            0 30px 60px -10px rgba(99, 102, 241, 0.2), 
            inset 0 0 0 1px rgba(255, 255, 255, 1);
    }
    
    .status-alert {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 1px solid rgba(253, 164, 175, 0.8);
        color: #e11d48;
        text-align: center;
        padding: 14px 24px;
        border-radius: 50px; /* Elegant pill shape */
        font-weight: 700;
        letter-spacing: 1px;
        font-size: 1.1rem;
        margin-top: 30px;
        box-shadow: 0 10px 25px -5px rgba(225, 29, 72, 0.25);
    }
    .status-clear {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid rgba(134, 239, 172, 0.8);
        color: #16a34a;
        text-align: center;
        padding: 14px 24px;
        border-radius: 50px; /* Elegant pill shape */
        font-weight: 700;
        letter-spacing: 1px;
        font-size: 1.1rem;
        margin-top: 30px;
        box-shadow: 0 10px 25px -5px rgba(22, 163, 74, 0.25);
    }
    
    .header-text {
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #9333ea 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -15px !important;
        letter-spacing: -2.5px;
        line-height: 1.2;
    }
    .sub-text {
        color: #64748b;
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 3.5rem;
        margin-top: 15px;
        letter-spacing: -0.2px;
    }
    
    .viewer {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 36px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        padding: 24px;
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.03);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
    }
    
    .detail-row {
        display: flex;
        justify-content: space-between;
        padding: 16px 0;
        border-bottom: 1px solid rgba(226, 232, 240, 0.6);
    }
    .detail-label {
        color: #64748b;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .detail-value {
        color: #0f172a;
        font-weight: 800;
        text-align: right;
        font-size: 1.1rem;
    }
    
    /* Elite Interactive Buttons */
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: #ffffff !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        box-shadow: 0 8px 16px -4px rgba(79, 70, 229, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #6366f1 0%, #60a5fa 100%);
        box-shadow: 0 12px 24px -6px rgba(79, 70, 229, 0.6);
        transform: translateY(-3px);
        color: #ffffff !important;
        border: none !important;
    }
    
    div.stButton > button:active, div.stDownloadButton > button:active {
        background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%);
        box-shadow: inset 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(1px);
        color: #e2e8f0 !important;
        border: none !important;
    }
    
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header-text'>Intelligent RTO Nexus ✨</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Real-time Visual Analytics & National Sentinel Database</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["✨ Sentinel Vision", "🛡️ National Registry", "📊 Operational Analytics"])

def record_to_db(results):
    """Saves fast-tracked, ultra-forgiving neural detections into memory."""
    for r in results:
        data = r.get("rto_data", {})
        plate = r.get("plate_text", "")
        if plate:
            # Even unreadable plates get logged now so the user knows it's working
            display_plate = "UNKNOWN PLATE" if plate == "UNREADABLE" else plate
            existing = [rec['License Plate'] for rec in st.session_state.db_records]
            if display_plate not in existing or display_plate == "UNKNOWN PLATE":
                st.session_state.db_records.append({
                    "License Plate": display_plate,
                    "Target Country": "India",
                    "State Region": data.get("state", "Unknown"),
                    "Owner": data.get("owner", "Unknown"),
                    "Type of Vehicle": data.get("type_of_vehicle", "Unknown"),
                    "Vehicle Model": data.get("model", "Unknown"),
                    "Registration Date": data.get("registration_date", "N/A"),
                    "Expiration Date": data.get("expiry", "N/A"),
                    "Status": "REQUIRES MANUAL REVIEW" if plate == "UNREADABLE" else data.get("status", "CLEARED"),
                })

def render_results(results):
    if not results:
        st.info("✨ Deep-scan array active. Awaiting visual confirmation of plate anomalies.")
        st.warning("⚠ No data extracted from image. Ensure the holding object is directly in the center.")
        return
        
    for res in results:
        data = res.get("rto_data", {})
        plate = res.get("plate_text", "UNREADABLE")
        status = data.get("status", "CLEARED")
        confidence = res.get("confidence", 0.0)
        
        display_plate = "⚠ OCR FAILED" if plate == "UNREADABLE" else plate
        
        html = f"""
        <div class="metric-box">
            <h3 style="margin-top:0; color:#0f172a; font-size:2.8rem; font-weight:800; letter-spacing:2px; text-align:center;">{display_plate}</h3>
            <p style="text-align:center; color:#64748b; font-size:1.05rem; margin-top:-15px; font-weight:600; margin-bottom: 30px;">Neural Confidence Match: <span style="color:#8b5cf6;">{confidence*100:.1f}%</span></p>
            
            <div class="detail-row">
                <span class="detail-label">Origin Nation</span>
                <span class="detail-value">India</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">State Region</span>
                <span class="detail-value">{data.get("state", "—")}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Classification</span>
                <span class="detail-value">{data.get("type_of_vehicle", "—")}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Vehicle Identity</span>
                <span class="detail-value">{data.get("model", "—")}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Registered Owner</span>
                <span class="detail-value">{data.get("owner", "—")}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Federal Expiry Date</span>
                <span class="detail-value">{data.get("expiry", "—")}</span>
            </div>
            <div class="detail-row" style="border:none;">
                <span class="detail-label">Driver Warrants</span>
                <span class="detail-value" style="color: {'#e11d48' if status != 'CLEARED' else '#16a34a'};">
                    {'⚠ PENDING' if status != 'CLEARED' else 'No Active Warrants'}
                </span>
            </div>
        """
        
        if status == "CLEARED":
            html += f'<div class="status-clear">✓ Clearance: {status}</div></div>'
        else:
            html += f'<div class="status-alert">⚠ Alert: {status}</div></div>'
            
        st.markdown(html, unsafe_allow_html=True)


# ==========================================================
# TAB 1: SURVEILLANCE DASHBOARD (Sentinel Vision)
# ==========================================================
with tab1:
    col1, col2 = st.columns([2.3, 1.2])

    with col1:
        st.markdown('<div class="viewer">', unsafe_allow_html=True)
        mode = st.radio("Vision Source Selector", ["Static Image Forensics", "Camera Intelligence", "Video Array"], horizontal=True)
        
        if mode == "Static Image Forensics":
            file = st.file_uploader("Upload visual evidence of target vehicle", type=["jpg", "jpeg", "png"])
            if file:
                image = Image.open(file).convert('RGB')
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                with st.spinner("✨ Quantum-array processing initialised... extracting alphanumeric structures..."):
                    # CRITICAL FIX: Ensure frame_number is 0 so the pipeline doesn't drop the frame!
                    annotated, results = pipeline.process(frame, frame_number=0, fallback_to_full_frame=True)
                    record_to_db(results)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)
                    with col2:
                        render_results(results)

        elif mode == "Camera Intelligence":
            st.info("Align target license plate within the optic sensor array bounds.")
            camera_pic = st.camera_input("Optical Sensor Feed")
            if camera_pic:
                image = Image.open(camera_pic).convert('RGB')
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                with st.spinner("✨ Intercepting plate geometrics... Applying ultra-loose fallback models..."):
                    # CRITICAL FIX: Ensure frame_number is 0 so the pipeline doesn't drop the frame!
                    annotated, results = pipeline.process(frame, frame_number=0, fallback_to_full_frame=True)
                    record_to_db(results)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)
                    with col2:
                        render_results(results)

        elif mode == "Video Array":
            video_file = st.file_uploader("Upload Traffic Corridor Feed", type=["mp4", "mov", "avi"])
            if video_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file.read())
                
                if st.button("▶️ Initialise Frame-by-Frame Deep Scan"):
                    stframe = st.empty()
                    status_pane = col2.empty()
                    
                    cap = cv2.VideoCapture(tfile.name)
                    frame_n = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frame_n += 1
                        
                        annotated, results = pipeline.process(frame, frame_number=frame_n, fallback_to_full_frame=False)
                        if results: record_to_db(results)
                        
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        stframe.image(annotated_rgb, channels="RGB")
                        
                        if results:
                            with status_pane.container():
                                render_results(results)
                    
                    cap.release()
                    st.success("✅ Surveillance operation concluded.")
                    
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================================
# TAB 2: GOVERNMENT DATABASE (National Registry)
# ==========================================================
with tab2:
    if "admin_logged" not in st.session_state:
        st.session_state.admin_logged = False
        
    if not st.session_state.admin_logged:
        st.markdown("<br><div class='metric-box' style='max-width:550px; margin:0 auto; text-align:center;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='font-size:2rem; font-weight:800;'>🛡️ National Intelligence Node</h2>", unsafe_allow_html=True)
        st.info("Federal Access Clearance Required.")
        pwd = st.text_input("Enter Biometric / Alpha-Numeric Passkey", type="password")
        if st.button("Authorise Connection", use_container_width=True):
            if pwd == "admin123":
                st.session_state.admin_logged = True
                st.rerun()
            else:
                st.error("Authentication Terminated. Security nodes deployed.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("### 🗂️ Identified Targets Matrix")
        
        col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
        with col_f1: 
            show_threats = st.checkbox("🔍 Isolate Anomalies (Expired/Unregistered Threats Only)")
            
        display_data = st.session_state.db_records
        if show_threats:
            display_data = [d for d in display_data if d["Status"] != "CLEARED"]
            
        if display_data:
            ordered_cols = ["License Plate", "Target Country", "State Region", "Vehicle Model", "Owner", "Registration Date", "Expiration Date", "Type of Vehicle", "Status"]
            df = pd.DataFrame(display_data)
            for col in ordered_cols:
                if col not in df.columns: df[col] = "—"
            df = df[ordered_cols]
            
            # Action button
            with col_f2:
                csv = df.to_csv(index=False)
                st.download_button("📥 Extract Intel (CSV)", csv, "rto_surveillance.csv", "text/csv", use_container_width=True)
            with col_f3:
                if st.button("🔒 Sever Connection", use_container_width=True):
                    st.session_state.admin_logged = False
                    st.rerun()
            
            def color_registry(val):
                color = '#e11d48' if val != 'CLEARED' else '#059669'
                weight = 'bold' if val != 'CLEARED' else '500'
                background = '#ffe4e6' if val != 'CLEARED' else 'transparent'
                return f'color: {color}; font-weight: {weight}; background-color: {background};'
            
            styled_df = df.style.map(color_registry, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True, height=500)
        else:
            with col_f3:
                if st.button("🔒 Sever Connection", use_container_width=True):
                    st.session_state.admin_logged = False
                    st.rerun()
            st.success("✨ Zero anomalies detected within current query parameters.")


# ==========================================================
# TAB 3: OPERATIONAL ANALYTICS
# ==========================================================
with tab3:
    st.markdown("### 📈 Active Area Intelligence")
    
    if len(st.session_state.db_records) == 0:
        st.warning("No data points available to construct intelligence graph.")
    else:
        df_stats = pd.DataFrame(st.session_state.db_records)
        
        c1, c2, c3 = st.columns(3)
        threat_count = len(df_stats[df_stats["Status"] != "CLEARED"])
        clear_count = len(df_stats[df_stats["Status"] == "CLEARED"])
        total_count = len(df_stats)
        
        c1.markdown(f"""
            <div class="metric-box" style="text-align:center;">
                <h3 style="color:#0f172a; margin-bottom:0;">Total Intercepts</h3>
                <h1 style="color:#3b82f6; font-size:3rem; margin-top:0;">{total_count}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        c2.markdown(f"""
            <div class="metric-box" style="text-align:center;">
                <h3 style="color:#dc2626; margin-bottom:0;">Active Threats</h3>
                <h1 style="color:#ef4444; font-size:3rem; margin-top:0;">{threat_count}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        c3.markdown(f"""
            <div class="metric-box" style="text-align:center;">
                <h3 style="color:#16a34a; margin-bottom:0;">Verified Cleared</h3>
                <h1 style="color:#22c55e; font-size:3rem; margin-top:0;">{clear_count}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        # Simple breakdown
        st.markdown("#### 📊 Vehicle Category Analytics")
        st.markdown("Distribution of vehicle classifications currently operating in the surveilled zone.")
        st.bar_chart(df_stats['Type of Vehicle'].value_counts())

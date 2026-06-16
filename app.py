import streamlit as st
import numpy as np
import cv2
import pandas as pd
import os
import random
import traceback
import math 
from PIL import Image

# --- CONFIGURATION & CONSTANTS ---
PAGE_TITLE = "Oil Spill Forensic System"
PAGE_ICON = "🛢️"

# --- SMART PATH FINDER ---
def find_file(filename, search_subdirs=["saved_models", "data/ais_data", "../saved_models", "../data/ais_data"]):
    """
    Searches for a file in common locations relative to app.py.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, filename),           
        os.path.join(base_dir, "..", filename),     
    ]
    for subdir in search_subdirs:
        candidates.append(os.path.join(base_dir, subdir, filename))
        
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None 

# Locate Models
unet_path = find_file("unet_oil_spill.h5") or "saved_models/unet_oil_spill.h5"
deeplab_path = find_file("deeplabv3_oil_spill.h5") or "saved_models/deeplabv3_oil_spill.h5"

MODEL_PATHS = {
    "UNet (Standard)": unet_path,
    "DeepLabV3+ (Experimental)": deeplab_path
}

# Locate Data
ais_path = find_file("vessel_data_clean.csv")
AIS_DATA_PATH = ais_path if ais_path else "data/ais_data/vessel_data_clean.csv"

TIPS_AND_TRICKS = [
    "💡 Tip: Higher sensitivity may increase false positives in choppy water.",
    "ℹ️ Note: Analysis assumes Sentinel-1 imagery resolution (~10m/pixel).",
    "🔍 Insight: Verify AIS correlation to identify potential offending vessels.",
    "🚀 Status: System optimization active for rapid inference.",
]

# --- 1. UI & CSS SETUP ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* DARK THEME & UI POLISH */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at top, #0f172a 0%, #020617 60%); background-attachment: fixed; color: #e5e7eb; }
header[data-testid="stHeader"] { background: transparent; height: 0px; }
footer { visibility: hidden; }
.glass-card { background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(18px); border-radius: 18px; border: 1px solid rgba(148, 163, 184, 0.15); padding: 24px; margin-bottom: 24px; }
h1 { color: #e5e7eb; font-weight: 800; letter-spacing: -1px; }
h3, h2 { color: #38bdf8; font-weight: 600; }
p, label, .stMarkdown, .stCaption { color: #cbd5f5; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #020617, #020617); border-right: 1px solid rgba(148,163,184,0.12); }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #e5e7eb; }
.stButton > button { border-radius: 10px; font-weight: 600; border: none; background-image: linear-gradient(135deg, #38bdf8, #2563eb); color: #020617; }
.severity-box { padding: 18px; border-radius: 14px; font-weight: 600; text-align: center; margin-top: 12px; }
.sev-critical { background: rgba(239, 68, 68, 0.15); color: #fecaca; border-left: 5px solid #ef4444; }
.sev-high { background: rgba(249, 115, 22, 0.15); color: #fed7aa; border-left: 5px solid #f97316; }
.sev-moderate { background: rgba(59, 130, 246, 0.18); color: #bfdbfe; border-left: 5px solid #3b82f6; }
.sev-clean { background: rgba(34, 197, 94, 0.15); color: #bbf7d0; border-left: 5px solid #22c55e; }
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND LOGIC INTEGRATION ---

# === NEW: Custom Metrics needed for Loading ===
# These must match what was used in training exactly
def dice_loss(y_true, y_pred, smooth=1e-6):
    import tensorflow.keras.backend as K
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return 1 - dice

def iou_metric(y_true, y_pred, smooth=1e-6):
    import tensorflow.keras.backend as K
    y_pred_metric = K.cast(K.greater(y_pred, 0.5), K.floatx())
    intersection = K.sum(K.abs(y_true * y_pred_metric), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3]) + K.sum(y_pred_metric,[1,2,3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)

def dice_coeff_metric(y_true, y_pred, smooth=1e-6):
    import tensorflow.keras.backend as K
    y_pred_metric = K.cast(K.greater(y_pred, 0.5), K.floatx())
    intersection = K.sum(y_true * y_pred_metric, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred_metric, axis=[1,2,3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

@st.cache_resource
def load_backend_model(path):
    """
    Loads the trained Keras model with custom objects.
    """
    try:
        if not os.path.exists(path):
            return None, "File path check failed (Code 404)"
        
        import tensorflow as tf
        
        # FIX: Pass the custom objects dictionary
        custom_objects_dict = {
            'dice_loss': dice_loss,
            'iou_metric': iou_metric,
            'dice_coeff_metric': dice_coeff_metric
        }
        
        model = tf.keras.models.load_model(path, custom_objects=custom_objects_dict)
        return model, None
    except Exception as e:
        return None, str(e)

def run_inference(model, image_input, threshold):
    """
    Performs image preprocessing, inference, and thresholding.
    """
    try:
        # Preprocessing
        img_resized = cv2.resize(image_input, (256, 256))
        img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
        
        # Inference
        raw_pred = model.predict(img_tensor, verbose=0)[0]
        
        # Handle dimensions
        if raw_pred.ndim == 3 and raw_pred.shape[2] == 1:
            raw_pred_2d = np.squeeze(raw_pred, axis=-1)
        else:
            raw_pred_2d = raw_pred

        # Thresholding
        mask = (raw_pred_2d > threshold).astype(np.uint8)
        
        return img_resized, mask, raw_pred_2d
    except Exception as e:
        raise e

def analyze_damage(mask, pixel_res_m2=100):
    oil_pixels = np.count_nonzero(mask)
    total_area_m2 = oil_pixels * pixel_res_m2
    total_area_km2 = total_area_m2 / 1_000_000.0
    
    severity = "NONE"
    css_class = "sev-clean"
    msg = "✅ NO SPILL DETECTED — Area is clear"

    if total_area_km2 > 1.0:
        severity = "CRITICAL"
        css_class = "sev-critical"
        msg = f"🚨 **CRITICAL** SEVERITY — Cleanup required ({total_area_km2:.2f} km²)"
    elif total_area_km2 > 0.1:
        severity = "HIGH"
        css_class = "sev-high"
        msg = f"⚠️ **HIGH** SEVERITY — Booms advised ({total_area_km2:.2f} km²)"
    elif total_area_km2 > 0.0:
        severity = "MODERATE"
        css_class = "sev-moderate"
        msg = f"ℹ️ **MODERATE** SEVERITY — Minor leakage ({total_area_km2:.4f} km²)"
        
    return {
        "pixels": oil_pixels,
        "area_km2": total_area_km2,
        "severity": severity,
        "css_class": css_class,
        "message": msg
    }

@st.cache_data
def load_ais_data():
    if os.path.exists(AIS_DATA_PATH):
        try:
            return pd.read_csv(AIS_DATA_PATH)
        except Exception:
            return None
    return None

def get_ais_anomalies(lat, lon, search_radius=2.0):
    try:
        df = load_ais_data()
        if df is not None:
            nearby = df[
                (df['LAT'] > lat - search_radius) & 
                (df['LAT'] < lat + search_radius) & 
                (df['LON'] > lon - search_radius) & 
                (df['LON'] < lon + search_radius)
            ]
            if nearby.empty: return []
            
            suspects = []
            for _, ship in nearby.iterrows():
                status = "STOPPED" if ship['SOG'] < 1.0 else "MOVING"
                suspects.append({
                    "name": str(ship['VesselName']),
                    "mmsi": ship['MMSI'],
                    "speed": ship['SOG'],
                    "status": status
                })
            return suspects
        else:
            return [
                {"name": "SIMULATED TANKER A", "mmsi": 999123456, "speed": 0.2, "status": "STOPPED"},
                {"name": "CARGO SHIP B", "mmsi": 888123456, "speed": 14.5, "status": "MOVING"},
            ]
    except Exception:
        return []

@st.cache_data
def load_rescue_stations():
    station_data = [
        {"Name": "USCG Sector New Orleans", "Lat": 29.95, "Lon": -90.07},
        {"Name": "Port Fourchon Rapid Response", "Lat": 29.17, "Lon": -90.20},
        {"Name": "Galveston Coast Guard Base", "Lat": 29.30, "Lon": -94.79},
        {"Name": "Mobile Bay USCG", "Lat": 30.27, "Lon": -88.00},
        {"Name": "Corpus Christi SAR", "Lat": 27.78, "Lon": -97.39},
    ]
    return pd.DataFrame(station_data)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * R

def find_nearest_station(spill_lat, spill_lon):
    stations_df = load_rescue_stations()
    if stations_df.empty: return None
    stations_df['Distance_km'] = stations_df.apply(
        lambda row: haversine_distance(spill_lat, spill_lon, row['Lat'], row['Lon']), axis=1
    )
    return stations_df.iloc[stations_df['Distance_km'].idxmin()]

def load_demo_image():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, "data/test/images"),
        os.path.join(base_dir, "../data/test/images"),
        os.path.join(base_dir, "data/train/images")
    ]
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for folder_path in possible_paths:
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions]
            if len(files) > 0:
                selected_file = random.choice(files)
                full_path = os.path.join(folder_path, selected_file)
                return full_path, selected_file
    return None, None

# --- 3. UI COMPONENTS ---

def render_sidebar():
    with st.sidebar:
        st.header("🔧 Control Panel")
        st.markdown("---")
        
        st.subheader("AI Configuration")
        selected_model_name = st.selectbox("Select Architecture", list(MODEL_PATHS.keys()), index=0)
        current_model_path = MODEL_PATHS[selected_model_name]

        st.subheader("Model Status")
        model, error = load_backend_model(current_model_path)
        
        if model:
            st.success(f"Loaded: {selected_model_name}")
        else:
            st.warning("Running in **DEMO MODE**")
            st.caption(f"Error: {error}") # Detailed error msg
            
        st.markdown("---")
        
        st.subheader("Input Data")
        upload = st.file_uploader("Upload SAR Image", type=['jpg', 'png', 'jpeg'])
        
        st.subheader("Parameters")
        threshold = st.slider("**Detection Sensitivity**", 0.0, 1.0, 0.05, 0.01)
        alpha = st.slider("**Overlay Opacity**", 0.1, 1.0, 0.6)
        
        st.info(random.choice(TIPS_AND_TRICKS))
        return upload, threshold, alpha, model

def main():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("🛢️ Oil Spill Forensic System")
    st.caption("Satellite SAR Analysis & Environmental Damage Assessment")
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file, threshold, alpha, model = render_sidebar()

    if 'demo_active' not in st.session_state: st.session_state['demo_active'] = False
        
    if uploaded_file is not None or st.session_state['demo_active']:
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            filename = uploaded_file.name
            st.session_state['demo_active'] = False 
        else:
            demo_path, demo_filename = load_demo_image()
            if demo_path and st.session_state['demo_active']:
                input_image = cv2.imread(demo_path)
                if input_image is not None:
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                    filename = f"Demo: {demo_filename}"
                else:
                    input_image = np.zeros((256, 256, 3), dtype=np.uint8)
                    filename = "Error reading demo image"
            else:
                input_image = np.zeros((256, 256, 3), dtype=np.uint8)
                filename = "No images found"
                st.session_state['demo_active'] = False

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🚀 Analysis Pipeline")
        
        if st.button("Run Forensic Analysis", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Preprocessing image...")
                progress_bar.progress(25)
                status_text.text("Running Neural Network Inference...")
                
                if model:
                    img_resized, mask, raw_prob = run_inference(model, input_image, threshold)
                else:
                    img_resized = cv2.resize(input_image, (256, 256))
                    raw_prob = np.zeros((256, 256), dtype=np.float32)
                    cv2.circle(raw_prob, (128,128), 40, 0.8, -1) 
                    mask = (raw_prob > threshold).astype(np.uint8)
                    
                progress_bar.progress(75)
                status_text.text("Calculating severity and checking AIS data...")
                damage_report = analyze_damage(mask)
                
                SPILL_LAT, SPILL_LON = 28.5, -90.5 
                suspects = get_ais_anomalies(SPILL_LAT, SPILL_LON) 
                nearest_station_data = find_nearest_station(SPILL_LAT, SPILL_LON)

                progress_bar.progress(100)
                status_text.empty()
                
                st.markdown(f'<div class="severity-box {damage_report["css_class"]}">{damage_report["message"]}</div>', unsafe_allow_html=True)
                st.write("")

                c1, c2, c3 = st.columns(3)
                mask_vis = np.zeros_like(img_resized)
                mask_vis[:, :, 0] = mask * 255 
                overlay = cv2.addWeighted(img_resized, 1.0, mask_vis, alpha, 0)

                with c1: st.image(img_resized, caption=f"Input: **{filename}**", use_container_width=True)
                with c2: st.image(mask * 255, caption="Binary Segmentation Mask", use_container_width=True)
                with c3: st.image(overlay, caption="Forensic Overlay (Red)", use_container_width=True)

                st.markdown("### 📊 Forensic Report")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Detected Oil Pixels", f"{damage_report['pixels']:,}")
                m2.metric("Spill Area", f"{damage_report['area_km2']:.4f} km²")
                m3.metric("Max Probability", f"{np.max(raw_prob)*100:.1f}%")
                m4.metric("Process Latency", "**Instant**")
                
                st.markdown("### 🗺️ Operational Infrastructure") 
                if nearest_station_data is not None:
                    map_data = pd.DataFrame({
                        'lat': [SPILL_LAT, nearest_station_data['Lat']],
                        'lon': [SPILL_LON, nearest_station_data['Lon']],
                        'color': ['#ff0000', '#0000ff'],
                    })
                    n1, n2 = st.columns([1, 2])
                    with n1:
                         st.metric(label="Closest Response Base", value=nearest_station_data['Name'], delta=f"{nearest_station_data['Distance_km']:.1f} km away")
                         st.caption(f"Coordinates: ({nearest_station_data['Lat']:.2f}, {nearest_station_data['Lon']:.2f})")
                    with n2:
                        st.caption("Map: Red pin is Spill, Blue pin is Nearest Station.")
                        st.map(map_data, latitude='lat', longitude='lon', color='color', zoom=8, use_container_width=True) 
                    st.write("---")
                else:
                    st.warning("Could not load rescue station data.")
                
                with st.expander("🚢 Nearby Vessel Activity (AIS Data)", expanded=True):
                    if suspects:
                        st.dataframe(pd.DataFrame(suspects)[['name', 'mmsi', 'speed', 'status']], use_container_width=True)
                        st.caption("⚠️ Vessels with '**STOPPED**' status near the spill coordinates are primary suspects.")
                    else:
                        st.info("No vessels detected in the immediate vicinity.")

                st.download_button("📥 Download Forensic Report (CSV)", pd.DataFrame([damage_report]).to_csv(index=False), "forensic_report.csv", "text/csv")

            except Exception as e:
                st.error("Analysis Failed")
                st.error(f"Error details: {e}")
                with st.expander("Debug Trace"): st.text(traceback.format_exc())
        
        else:
            st.info("👈 Upload an image and click 'Run Forensic Analysis' to start.")
            col_ph1, col_ph2 = st.columns(2)
            with col_ph1: st.image(input_image, caption=f"Preview: **{filename}**", width=300)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card" style="text-align: center; padding: 60px;">', unsafe_allow_html=True)
        st.subheader("Ready for Analysis")
        st.markdown("Please upload a Sentinel-1 SAR image from the sidebar.")
        if st.button("🎲 Load Demo Data"):
            st.session_state['demo_active'] = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
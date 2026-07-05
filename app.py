import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import numpy as np
import av
import cv2
import csv
import os
import time
import threading
from datetime import datetime

# =================================================================
# Page setup (matches report: "Streamlit Page Setup" component)
# =================================================================
st.set_page_config(page_title="TATASecure Vision", page_icon="🛡️", layout="wide")

CONF_THRESH = 0.25
COOLDOWN = 5  # seconds, same as the original script

PERSON_LABEL = "human"
HELMET_LABEL = "helmet"
VEST_LABEL = "vest"

LOG_FILE = "violations.csv"
SCREENSHOT_DIR = "violation_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Violation"])

# Danger zone polygon (Region Compliance module, Shapely-based)
danger_zone = Polygon([(100, 100), (500, 100), (500, 400), (100, 400)])
zone_points = np.array([(100, 100), (500, 100), (500, 400), (100, 400)])


def normalize(label: str) -> str:
    return label.lower().replace(" ", "-").replace("_", "-")


def log_violation(violation_type: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([timestamp, violation_type])


def save_screenshot(frame_img, violation_type: str):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_type = violation_type.lower().replace(" ", "_")
    filename = f"{safe_type}_{ts}.jpg"
    path = os.path.join(SCREENSHOT_DIR, filename)
    cv2.imwrite(path, frame_img)
    return path


# =================================================================
# Model loading (cached so it only loads once per session)
# =================================================================
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")


model = load_model()


# =================================================================
# Video processor — runs the inference + violation logic per frame
# =================================================================
class PPEVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.last_helmet_log = 0
        self.last_vest_log = 0
        self.last_zone_log = 0
        self.live_counts = {"persons": 0, "helmet_violations": 0, "vest_violations": 0}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=CONF_THRESH)[0]
        annotated = results.plot()

        persons = 0
        helmets = 0
        vests = 0
        now = time.time()

        for box in results.boxes:
            label = normalize(model.names[int(box.cls[0])])

            if label == PERSON_LABEL:
                persons += 1
                x1, y1, x2, y2 = box.xyxy[0]
                center = Point(int((x1 + x2) / 2), int((y1 + y2) / 2))
                if danger_zone.contains(center):
                    cv2.putText(annotated, "ZONE VIOLATION", (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    with self.lock:
                        if now - self.last_zone_log > COOLDOWN:
                            log_violation("ZONE VIOLATION")
                            save_screenshot(annotated, "ZONE VIOLATION")
                            self.last_zone_log = now

            elif label == HELMET_LABEL:
                helmets += 1
            elif label == VEST_LABEL:
                vests += 1

        # No dedicated "no-helmet"/"no-vest" classes in this model, so
        # violations are inferred by shortage: more people than PPE items
        helmet_violation = persons > helmets
        vest_violation = persons > vests

        if helmet_violation:
            cv2.putText(annotated, "HELMET VIOLATION", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            with self.lock:
                if now - self.last_helmet_log > COOLDOWN:
                    log_violation("HELMET VIOLATION")
                    save_screenshot(annotated, "HELMET VIOLATION")
                    self.last_helmet_log = now

        if vest_violation:
            cv2.putText(annotated, "VEST VIOLATION", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            with self.lock:
                if now - self.last_vest_log > COOLDOWN:
                    log_violation("VEST VIOLATION")
                    save_screenshot(annotated, "VEST VIOLATION")
                    self.last_vest_log = now

        cv2.polylines(annotated, [zone_points], True, (0, 0, 255), 2)

        with self.lock:
            self.live_counts = {
                "persons": persons,
                "helmet_violations": int(helmet_violation),
                "vest_violations": int(vest_violation),
            }

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# =================================================================
# UI layout
# =================================================================
st.markdown(
    "<h1 style='text-align:center;'>TATA<span style='color:#F5A623;'>Secure</span> Vision</h1>"
    "<p style='text-align:center; color:gray;'>Smart vision safeguarding workforce safety in real time</p>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live PPE & Zone Monitoring")
    ctx = webrtc_streamer(
        key="ppe-detection",
        video_processor_factory=PPEVideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Live Status")

    # webrtc runs frame processing on a background thread, so the main
    # script needs to rerun periodically to pick up updated live_counts —
    # otherwise this panel stays frozen at its initial values.
    st_autorefresh(interval=1000, key="status_refresh")

    status_box = st.empty()

    if ctx.video_processor:
        counts = ctx.video_processor.live_counts
        status_box.metric("Persons Detected", counts["persons"])
        st.metric("Helmet Violations", counts["helmet_violations"])
        st.metric("Vest Violations", counts["vest_violations"])
    else:
        status_box.info("Start the camera to see live detections.")

    st.subheader("Recent Violation Log")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            raw_rows = list(csv.reader(f))[1:]  # skip header

        # Defensive parsing: only keep rows with at least 2 columns,
        # and only use the first two (timestamp, violation) even if
        # a row has extra stray columns from an older log format.
        rows = [row for row in raw_rows if len(row) >= 2]

        recent = rows[-10:][::-1]
        if recent:
            for row in recent:
                ts, violation = row[0], row[1]
                st.write(f"🔴 **{violation}** — {ts}")
        else:
            st.write("No violations logged yet.")

        skipped = len(raw_rows) - len(rows)
        if skipped > 0:
            st.caption(f"⚠️ Skipped {skipped} malformed row(s) in {LOG_FILE}.")

    st.subheader("Latest Snapshot")
    if os.path.exists(SCREENSHOT_DIR):
        shots = sorted(os.listdir(SCREENSHOT_DIR))
        if shots:
            st.image(os.path.join(SCREENSHOT_DIR, shots[-1]), caption=shots[-1])
        else:
            st.write("No screenshots saved yet.")
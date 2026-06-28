from ultralytics import YOLO
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
import csv
from datetime import datetime
import os
import time

# Load model
model = YOLO("best.pt")
names = model.names
COOLDOWN = 5  # seconds

last_helmet_log = 0
last_vest_log = 0
last_zone_log = 0

# Open webcam
cap = cv2.VideoCapture(0)

# CSV log file
log_file = "violations.csv"

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Violation"])

# Logging function
def log_violation(violation_type):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, violation_type])

# Danger Zone Coordinates
danger_zone = Polygon([
    (100, 100),
    (500, 100),
    (500, 400),
    (100, 400)
])

zone_points = np.array([
    (100, 100),
    (500, 100),
    (500, 400),
    (100, 400)
])

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    persons = 0
    helmets = 0
    vests = 0

    for box in results[0].boxes:

        label = names[int(box.cls[0])]
        conf = float(box.conf[0])

        print(f"{label}: {conf:.2f}")

        if label == "human":
            persons += 1

        elif label == "helmet":
            helmets += 1

        elif label == "vest":
            vests += 1

        # Zone violation detection
        if label == "human":

            x1, y1, x2, y2 = box.xyxy[0]

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            point = Point(center_x, center_y)

            if danger_zone.contains(point):

                cv2.putText(
                    annotated,
                    "ZONE VIOLATION",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                current_time = time.time()

                if current_time - last_zone_log > COOLDOWN:
                    log_violation("ZONE VIOLATION")
                    last_zone_log = current_time

    # Draw danger zone
    cv2.polylines(
        annotated,
        [zone_points],
        True,
        (0, 0, 255),
        2
    )

    # Helmet violation
    if persons > helmets:

        cv2.putText(
            annotated,
            "HELMET VIOLATION",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        current_time = time.time()

        if current_time - last_helmet_log > COOLDOWN:
            log_violation("HELMET VIOLATION")
            last_helmet_log = current_time

    # Vest violation
    if persons > vests:

        cv2.putText(
            annotated,
            "VEST VIOLATION",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        current_time = time.time()

        if current_time - last_vest_log > COOLDOWN:
            log_violation("VEST VIOLATION")
            last_vest_log = current_time

    # Display counts
    print(f"Persons: {persons}")
    print(f"Helmets: {helmets}")
    print(f"Vests: {vests}")

    cv2.imshow("Smart Safety AI", annotated)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
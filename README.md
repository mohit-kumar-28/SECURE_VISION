# Secure Vision

AI-powered industrial safety monitoring system using YOLOv11 for PPE detection
and Shapely-based zone monitoring for restricted-area compliance.

## Project Structure
```
SmartSafetyAI/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── best.pt
│   └── yolo11s.pt          (auto-downloads on first run)
├── violation_images/        (auto-created)
├── violations.csv           (auto-created)
├── training/
│   └── datasets
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Place your trained PPE weights at `models/ppe_best.pt`.

3. Run the app:
   ```
   streamlit run app.py
   ```

4. Open the browser tab it launches, select a model from the sidebar
   (PPE Detection or Region Compliance), and click Start on the webcam
   widget.

## Models

- **PPE Detection**: Custom-trained YOLOv11 model detecting Helmet, Vest,
  Boots, Glasses (and their "No X" absence classes). Trained on a Roboflow
  PPE dataset.
- **Region Compliance**: Uses the pretrained COCO YOLOv11 model to detect
  people, then flags violations when a person's position falls inside a
  configurable danger zone (Shapely polygon check).

## Notes

- Violation events are logged to `violations.csv` and a snapshot is saved
  to `violation_images/`, with a 15-second cooldown per violation type to
  avoid duplicate spam.
- Harness Detection and Kitchen Safety (Kadhai) modules were designed and
  documented but not trained for this deployment — see the project report
  for the methodology and planned dataset sourcing for future work.

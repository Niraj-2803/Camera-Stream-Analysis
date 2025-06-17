import argparse
import json
import logging
from collections import defaultdict

import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO


# ----------------------------- Utility Functions ----------------------------- #

def compute_angle(a, b, c):
    """Return the angle ABC (at point *b*) in degrees."""
    a, b, c = map(np.array, (a, b, c))
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return np.nan
    cos = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def classify_posture(kp, conf, angle_info, img_h=None, min_visible=8):
    """Classify pose as *Standing*, *Sitting*, or *Uncertain*."""
    legs_ok = all(conf[i] > 0.3 for i in (11, 12, 13, 14, 15, 16))
    if legs_ok:
        knee = np.nanmean([angle_info['l_knee'], angle_info['r_knee']])
        hip  = np.nanmean([angle_info['l_hip'],  angle_info['r_hip']])
        if knee < 120 or hip < 120:
            return 'Sitting'
        if knee > 150 and hip > 150:
            return 'Standing'
        return 'Uncertain'

    if img_h is not None and np.sum(conf > 0.3) >= min_visible:
        shoulder_y = np.nanmean([kp[5][1], kp[6][1]])
        hip_y      = np.nanmean([kp[11][1], kp[12][1]])
        mid_y      = 0.5 * (shoulder_y + hip_y)
        return 'Sitting' if mid_y > img_h * 0.55 else 'Standing'

    return 'Uncertain'


SKEL = [  # 12 segments
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


# ------------------------------- Drawing ------------------------------------- #

def draw_pose(img, kp, conf, label, box=None):
    """Draw skeleton and a label (posture + timer) onto frame."""
    h, w = img.shape[:2]
    margin = 5
    # Bones
    for i, j in SKEL:
        if conf[i] > 0.3 and conf[j] > 0.3:
            xi, yi = map(int, kp[i]); xj, yj = map(int, kp[j])
            if margin < xi < w - margin and margin < yi < h - margin and \
               margin < xj < w - margin and margin < yj < h - margin:
                cv2.line(img, (xi, yi), (xj, yj), (0, 255, 0), 2)
    # Joints
    for (x, y), c in zip(kp, conf):
        if c > 0.3 and margin < x < w - margin and margin < y < h - margin:
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
    # Label
    if box is not None and label:
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(img, (x0, y0 - 18), (x1, y0), (0, 0, 0), -1)
        cv2.putText(img, label, (x0 + 2, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def draw_zones(img, zones, color=(255, 0, 0)):
    """Outline each polygonal zone on the frame."""
    for pts in zones.values():
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True, color=color, thickness=2)
    return img


# ----------------------------- Main Pipeline --------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Zone-based dwell & posture tracker (now with 'uncertain' time)")
    parser.add_argument("--model", required=True, help="YOLO pose model path (e.g. yolo11m-pose.pt)")
    parser.add_argument("--source", required=True, help="Input video")
    parser.add_argument("--output", default="out.mp4", help="Annotated video output path")
    parser.add_argument("--stats", default="stats.json", help="Path to save JSON stats")
    parser.add_argument("--show", action="store_true", help="Show live imshow window")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N frames")
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Zones definition (edit as needed)
    zones = {
        "zone_1": [
        (343.20512820512823, 368.974358974359),
        (507.3076923076923, 275.38461538461536),
        (431.6666666666667, 157.43589743589746),
        (235.51282051282055, 222.82051282051282)
      ],
      "zone_2": [
        (348.33333333333337, 374.1025641025641),
        (517.5641025641025, 290.7692307692307),
        (621.4102564102565, 438.2051282051282),
        (448.33333333333337, 533.0769230769231)
      ],
      "zone_3": [
        (463.71794871794873, 563.8461538461538),
        (670.1282051282052, 501.025641025641),
        (804.0916132478633, 719.0),
        (522.6923076923077, 719.0)
      ],
      "zone_5": [
        (665.0, 353.58974358974353),
        (838.0769230769231, 238.20512820512818),
        (1011.1538461538462, 427.9487179487179),
        (811.1538461538462, 574.1025641025641)
      ],
      "zone_4": [
        (818.8461538461538, 575.3846153846154),
        (1025.2564102564104, 429.23076923076917),
        (1250.897435897436, 594.6153846153846),
        (1137.202123397436, 719.0),
        (843.2051282051282, 719.0),
        (770.1282051282052, 617.6923076923076)
      ]

        # Add more zones if required
    }
    zone_polys = {name: Polygon(pts) for name, pts in zones.items()}

    # Stats structure – now includes "uncertain"
    stats = {
        z: defaultdict(lambda: {
            "dwell": 0.0,
            "standing": 0.0,
            "sitting": 0.0,
            "uncertain": 0.0,
        })
        for z in zones
    }

    # Video capture for FPS & size
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        logging.error(f"Cannot open video source {args.source}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        logging.warning("FPS unavailable; defaulting to 25")
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    dt = 1.0 / fps
    logging.info(f"Video opened (fps={fps}, size={w}x{h})")

    # Writer
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Load model
    logging.info(f"Loading model {args.model} …")
    model = YOLO(args.model)

    # Frame processing
    frame_idx = 0
    for res in model.track(source=args.source, stream=True, verbose=False):

        frame_idx += 1
        # bgr = cv2.cvtColor(res.orig_img, cv2.COLOR_RGB2BGR)
        bgr = res.orig_img.copy()        # already in BGR order


        try:
            kps = res.keypoints.xy.cpu().numpy()           # [N,17,2]
            confs = res.keypoints.conf.cpu().numpy()       # [N,17]
            boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, 'xyxy') else None
            ids = res.boxes.id.cpu().numpy() if hasattr(res.boxes, 'id') else np.arange(len(kps))
        except Exception as e:
            logging.warning(f"Frame {frame_idx}: keypoint extraction failed: {e}")
            continue

        # Draw zones once per frame
        draw_zones(bgr, zones)

        for i, kp in enumerate(kps):
            conf = confs[i]
            tid = int(ids[i])

            # Angle calculations
            ang = {k: np.nan for k in ("l_knee", "r_knee", "l_hip", "r_hip")}
            if conf[11] > 0.3 and conf[13] > 0.3 and conf[15] > 0.3:
                ang["l_knee"] = compute_angle(kp[11], kp[13], kp[15])
            if conf[12] > 0.3 and conf[14] > 0.3 and conf[16] > 0.3:
                ang["r_knee"] = compute_angle(kp[12], kp[14], kp[16])
            if conf[5] > 0.3 and conf[11] > 0.3 and conf[13] > 0.3:
                ang["l_hip"] = compute_angle(kp[5], kp[11], kp[13])
            if conf[6] > 0.3 and conf[12] > 0.3 and conf[14] > 0.3:
                ang["r_hip"] = compute_angle(kp[6], kp[12], kp[14])

            posture = classify_posture(kp, conf, ang, img_h=h)

            # Determine zones
            x0, y0, x1, y1 = boxes[i] if boxes is not None else (0, 0, 0, 0)
            centroid = Point((x0 + x1) / 2, (y0 + y1) / 2)
            # draw centroid on the frame  (4‑pixel radius, cyan)
            cv2.circle(bgr, (int(centroid.x), int(centroid.y)), 4, (255, 255, 0), -1)
            
            inside = [z for z, poly in zone_polys.items() if poly.contains(centroid)]

            for z in inside:
                s = stats[z][tid]  # shorthand
                s["dwell"] += dt
                if posture.startswith('Sit'):
                    s['sitting'] += dt
                elif posture.startswith('Stand'):
                    s['standing'] += dt
                else:
                    s['uncertain'] += dt

            # Draw pose
            # Build label with timer if in a zone (using first zone found)
            label = posture
            if inside:
                dz = inside[0]
                dwell_sec = stats[dz][tid]["dwell"]
                label = f"{posture} {dwell_sec:0.1f}s"

            draw_pose(bgr, kp, conf, label, box=boxes[i] if boxes is not None else None)

        writer.write(bgr)
        if args.show:
            cv2.imshow("Zone Posture Tracker", bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("'q' pressed – exiting early …")
                break

        if frame_idx % args.log_every == 0:
            logging.info(f"Processed {frame_idx} frames (detections={len(kps)})")

    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # Report
    report = {"zones": {}}
    for z, persons in stats.items():
        report["zones"][z] = {}
        for tid, t in persons.items():
            report["zones"][z][f"person_{tid}"] = {
                "dwell_time_s":     round(t["dwell"],     2),
                "standing_time_s":  round(t["standing"],  2),
                "sitting_time_s":   round(t["sitting"],   2),
                "uncertain_time_s": round(t["uncertain"], 2),
            }
    # Save to JSON file
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Stats saved to {args.stats}")

    # Also print to stdout
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

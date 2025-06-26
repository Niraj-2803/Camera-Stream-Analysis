import cv2
import numpy as np
from ultralytics import YOLO

def blur_faces(frame, results, blur_size=35):

    result = results[0]  # single-image inference

    # (n_people,17,2) and (n_people,17)
    kpts  = result.keypoints.xy.cpu().numpy()
    confs = result.keypoints.conf.cpu().numpy()

    # make sure blur kernel is odd
    k = blur_size if blur_size % 2 == 1 else blur_size + 1

    h_img, w_img = frame.shape[:2]
    for person_kpts, person_conf in zip(kpts, confs):
        head_pts  = person_kpts[[0,1,2,3,4], :]
        head_conf = person_conf[[0,1,2,3,4]]
        valid = head_conf > 0.7
        pts = head_pts[valid]
        if pts.shape[0] < 2:
            continue

        xs, ys = pts[:,0], pts[:,1]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        # pad
        pw = int((x2 - x1) * 0.3)
        ph = int((y2 - y1) * 0.3)
        x1_p, y1_p = max(0, x1 - pw), max(0, y1 - ph)
        x2_p = min(w_img, x2 + pw)
        y2_p = min(h_img, y2 + ph)

        # square-ify
        w_box, h_box = x2_p - x1_p, y2_p - y1_p
        side = max(w_box, h_box)
        x2_s = min(w_img, x1_p + side)
        y2_s = min(h_img, y1_p + side)

        roi = frame[y1_p:y2_s, x1_p:x2_s]
        if roi.size > 0:
            frame[y1_p:y2_s, x1_p:x2_s] = cv2.blur(roi, (k, k))

    return frame


def main():
    # Load pose model
    model = YOLO('yolo11n-pose.pt')

    # Read image
    frame = cv2.imread(r'D:\All projects\Camex\image.png')

    # Run pose inference
    results = model(frame)

    # Draw keypoints + face boxes
    output = blur_faces(frame, results)

    # Save result
    out_path = 'frame_keypoints_boxes.jpg'
    cv2.imwrite(out_path, output)
    print(f"✅ Wrote output with keypoints and face rectangles to {out_path}")

if __name__ == '__main__':
    main()

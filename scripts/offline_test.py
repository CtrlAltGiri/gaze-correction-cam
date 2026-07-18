"""Offline pipeline test: run gaze correction on a still image with forced angles.

Usage: python scripts/offline_test.py <portrait.jpg> [output_dir]
Verifies the full detection + correction pipeline without a camera.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from displayers.face_predictor import create_face_predictor, EyeExtractionConfig
from model_managers.gaze_corrector_v1 import GazeCorrector

img_path = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(img_path) or "."
os.makedirs(out_dir, exist_ok=True)
frame = cv2.imread(img_path)
assert frame is not None, f"cannot read {img_path}"
h, w = frame.shape[:2]
print("frame", w, h)

pred = create_face_predictor("mediapipe")
gc = GazeCorrector()
cfg = EyeExtractionConfig()

faces = pred.list_eye_data(frame, cfg)
print("faces detected:", len(faces))
assert faces, "no face detected"
fd = faces[0]
print("eye centers:", fd.left_eye.center, fd.right_eye.center)

# natural correction (geometry-estimated)
out_natural = gc.apply_correction(frame.copy(), fd, (w, h))
alpha, eyepos = gc.estimate_gaze_angle(fd.left_eye.center, fd.right_eye.center, (w, h))
print("estimated alpha (v,h):", alpha, "eye pos:", [round(x,1) for x in eyepos])
cv2.imwrite(os.path.join(out_dir, "out_natural.png"), out_natural)

# forced angles to make effect obvious
for name, ang in [("up20", [20, 0]), ("down20", [-20, 0]), ("left20", [0, -20]), ("right20", [0, 20])]:
    f2 = frame.copy()
    le_c = gc.correct_eye(fd.left_eye, "L", ang)
    re_c = gc.correct_eye(fd.right_eye, "R", ang)
    pc = gc.pixel_cut
    for e, ec in [(fd.left_eye, le_c), (fd.right_eye, re_c)]:
        f2[e.top_left[0]+pc[0]:e.top_left[0]+e.original_size[0]-pc[0],
           e.top_left[1]+pc[1]:e.top_left[1]+e.original_size[1]-pc[1]] = ec[pc[0]:-pc[0], pc[1]:-pc[1]] * 255
    cv2.imwrite(os.path.join(out_dir, f"out_{name}.png"), f2)
    diff = np.abs(f2.astype(int) - frame.astype(int)).sum()
    print(name, "total abs diff:", diff)
print("done")

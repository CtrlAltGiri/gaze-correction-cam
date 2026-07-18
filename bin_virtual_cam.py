#!/usr/bin/env python3
"""
Virtual Camera Gaze Correction Application

Captures the real webcam, applies gaze correction, and outputs the corrected
feed to a virtual camera (OBS Virtual Camera on macOS) so it can be selected
in Zoom, Meet, Teams, FaceTime, and any other video app.

Requirements:
    - OBS Studio installed (provides the virtual camera device on macOS).
      Start OBS once and click "Start Virtual Camera" so the device is
      registered with the system, then quit OBS.
    - pip install pyvirtualcam

Usage:
    python bin_virtual_cam.py                       # mediapipe backend, camera 0
    python bin_virtual_cam.py --backend dlib        # dlib backend
    python bin_virtual_cam.py --camera 1            # camera device 1
    python bin_virtual_cam.py --preview             # also show a preview window

Controls (preview window focused):
    - 'g': Toggle gaze correction on/off
    - 'q': Quit
"""

import argparse
import time

import cv2

from displayers.dis_single_window import DisplayConfig
from displayers.face_predictor import EyeExtractionConfig, create_face_predictor
from model_managers.gaze_corrector_v1 import GazeCorrector
from utils.logger import Logger


def detect_camera_resolution(camera_id: int) -> tuple[int, int]:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Warning: Could not open camera {camera_id}, using default resolution")
        return (640, 480)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (width, height)


def main():
    parser = argparse.ArgumentParser(description="Gaze Correction Virtual Camera")
    parser.add_argument("--backend", type=str, default="mediapipe", choices=["dlib", "mediapipe"])
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument(
        "--config",
        type=str,
        default="./model_managers/gaze_corrector_v1_01.yaml",
        help="Path to gaze corrector config file",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output FPS")
    parser.add_argument("--preview", action="store_true", help="Show a preview window")
    args = parser.parse_args()

    try:
        import pyvirtualcam
    except ImportError:
        raise SystemExit(
            "pyvirtualcam is required: pip install pyvirtualcam\n"
            "On macOS also install OBS Studio and start its virtual camera once."
        )

    logger = Logger("VirtualCam")
    video_size = detect_camera_resolution(args.camera)
    logger.log(f"Camera resolution: {video_size[0]}x{video_size[1]}")

    predictor = create_face_predictor(args.backend)
    corrector = GazeCorrector(config_path=args.config)
    eye_config = EyeExtractionConfig()
    display_cfg = DisplayConfig(video_size=video_size)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])

    enabled = True
    frame_count = 0
    corrected_count = 0
    last_report = time.time()

    with pyvirtualcam.Camera(
        width=video_size[0], height=video_size[1], fps=args.fps, print_fps=False
    ) as vcam:
        logger.log(f"Virtual camera started: {vcam.device}")
        logger.log("Select this device in your video call app. Press Ctrl+C to stop.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.log("Failed to read frame")
                    break

                out = frame
                if enabled:
                    face_data_list = predictor.list_eye_data(frame, eye_config)
                    for face_data in face_data_list:
                        try:
                            out = corrector.apply_correction(
                                frame, face_data, display_cfg.video_size
                            )
                            corrected_count += 1
                        except Exception as e:
                            logger.log(f"Correction error: {e}")
                        break

                vcam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                frame_count += 1

                now = time.time()
                if now - last_report >= 10:
                    logger.log(
                        f"Frames: {frame_count}, corrected: {corrected_count} "
                        f"({100 * corrected_count / max(frame_count, 1):.0f}%)"
                    )
                    last_report = now

                if args.preview:
                    status = "GAZE ON" if enabled else "GAZE OFF"
                    color = (0, 255, 0) if enabled else (0, 0, 255)
                    disp = out.copy()
                    cv2.putText(
                        disp, status, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                    )
                    cv2.imshow("Gaze Correction (virtual cam)", disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("g"):
                        enabled = not enabled
                        logger.log(f"Gaze correction: {'enabled' if enabled else 'disabled'}")

                vcam.sleep_until_next_frame()
        except KeyboardInterrupt:
            logger.log("Interrupted")

    cap.release()
    if args.preview:
        cv2.destroyAllWindows()
    corrector.close()
    logger.log("Shutdown complete")


if __name__ == "__main__":
    main()

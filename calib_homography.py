import cv2
import numpy as np
import argparse
from utils.display import make_fullscreen_window

def on_mouse(event, x, y, flags, param):
    """
    Mouse callback: record up to 4 clicks as camera-space points.
    Points are in the camera frame's pixel coordinates.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        pts = param["pts"]
        if len(pts) < 4:
            pts.append((x, y))
            print(f"[calib] clicked CAMERA point #{len(pts)}: ({x}, {y})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_index", type=int, default=0,
                    help="Camera index (0 = default webcam)")
    ap.add_argument("--cam_w", type=int, default=640,
                    help="Camera capture width")
    ap.add_argument("--cam_h", type=int, default=480,
                    help="Camera capture height")
    ap.add_argument("--proj_w", type=int, default=1280,
                    help="Projector (or screen) width in pixels")
    ap.add_argument("--proj_h", type=int, default=720,
                    help="Projector (or screen) height in pixels")
    ap.add_argument("--out", default="homography.npy",
                    help="Output .npy file for 3x3 homography matrix")
    args = ap.parse_args()

    # ---------- PROJECTOR SIDE ----------
    # Projector canvas (all black with thin white border)
    proj = np.zeros((args.proj_h, args.proj_w, 3), np.uint8)
    cv2.rectangle(
        proj,
        (2, 2),
        (args.proj_w - 3, args.proj_h - 3),
        (255, 255, 255),
        2
    )

    # Fullscreen projector window on the target display.
    # NOTE: proj_w/proj_h should match the physical resolution of that display.
    make_fullscreen_window("PROJECTOR", x=0, y=0, fullscreen=True)
    cv2.imshow("PROJECTOR", proj)
    cv2.waitKey(1)

    # ---------- CAMERA SIDE ----------
    cap = cv2.VideoCapture(args.cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    # Explicitly set camera resolution so it matches your live script
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)

    # Grab one frame to confirm actual resolution
    ok, frame = cap.read()
    if not ok:
        cap.release()
        cv2.destroyAllWindows()
        raise RuntimeError("Failed to read from camera for calibration")

    h_cam, w_cam = frame.shape[:2]
    print(f"[calib] camera frame shape: {w_cam}x{h_cam} (W x H)")
    print("[calib] IMPORTANT: use the same cam_w/cam_h in your live script,")
    print("        and do NOT resize the CAMERA window during calibration.\n")

    # Set up CAMERA window and mouse callback
    click_state = {"pts": []}
    cv2.namedWindow("CAMERA", cv2.WINDOW_AUTOSIZE)  # 1:1 pixel mapping
    cv2.setMouseCallback("CAMERA", on_mouse, click_state)

    print("Click the 4 PROJECTOR rectangle corners in CAMERA view,")
    print("in this order: top-left, top-right, bottom-right, bottom-left.")
    print("Press ESC to abort.\n")

    # We already read one frame; show it first, then continue reading in a loop
    while True:
        # Use latest frame each iteration
        if not ok:
            ok, frame = cap.read()
            if not ok:
                print("[calib] camera read failed; exiting.")
                break

        # Draw already clicked points for feedback
        for (x, y) in click_state["pts"]:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("CAMERA", frame)

        k = cv2.waitKey(10) & 0xFF
        if k == 27:  # ESC to abort
            print("[calib] Aborted by user (ESC).")
            cap.release()
            cv2.destroyAllWindows()
            return

        if len(click_state["pts"]) == 4:
            # We have 4 camera points, now build correspondences
            cam_pts = np.array(click_state["pts"], dtype=np.float32)

            # Projector corners: full image rectangle
            proj_pts = np.array([
                [0, 0],
                [args.proj_w - 1, 0],
                [args.proj_w - 1, args.proj_h - 1],
                [0, args.proj_h - 1]
            ], dtype=np.float32)

            print("[calib] CAMERA points:\n", cam_pts)
            print("[calib] PROJECTOR points:\n", proj_pts)

            # Compute homography: camera -> projector
            H, mask = cv2.findHomography(cam_pts, proj_pts, method=0)
            if H is None:
                print("[calib] ERROR: Homography computation failed.")
            else:
                np.save(args.out, H)
                print(f"[calib] Saved homography (3x3) to {args.out}")
                print(H)
            break

        # After first display, continue reading next frame
        ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

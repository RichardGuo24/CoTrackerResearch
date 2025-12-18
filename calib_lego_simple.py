import cv2
import numpy as np
import argparse

def on_mouse(event, x, y, flags, param):
    """
    Mouse callback: record up to 4 clicks as camera-space points.
    Order: top-left, top-right, bottom-right, bottom-left.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        pts = param["pts"]
        if len(pts) < 4:
            pts.append((float(x), float(y)))
            print(f"[calib] clicked CAMERA corner #{len(pts)} at ({x}, {y})")

def make_projector_window(name, proj_w, proj_h, offset_x=0, offset_y=0, fullscreen=True):
    """
    Create and position the projector window on the projector display.
    offset_x/offset_y put the window on the correct monitor.
    Example: if main monitor is 2560x1440 on the left and projector is to the right,
    use offset_x=2560, offset_y=0.
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, offset_x, offset_y)
    cv2.resizeWindow(name, proj_w, proj_h)
    if fullscreen:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_index", type=int, default=0,
                    help="Camera index (0 = default)")
    ap.add_argument("--cam_w", type=int, default=1920,
                    help="Camera capture width (must match live script)")
    ap.add_argument("--cam_h", type=int, default=1080,
                    help="Camera capture height (must match live script)")
    ap.add_argument("--proj_w", type=int, default=1920,
                    help="Projector resolution width in pixels")
    ap.add_argument("--proj_h", type=int, default=1080,
                    help="Projector resolution height in pixels")
    ap.add_argument("--proj_offset_x", type=int, default=2560,
                    help="X offset to move projector window onto projector display")
    ap.add_argument("--proj_offset_y", type=int, default=0,
                    help="Y offset to move projector window onto projector display")
    ap.add_argument("--out", type=str, default="homography.npy",
                    help="Output .npy file for 3x3 homography matrix")
    args = ap.parse_args()

    # ---------- PROJECTOR SIDE ----------
    proj_name = "PROJECTOR"
    make_projector_window(
        proj_name,
        proj_w=args.proj_w,
        proj_h=args.proj_h,
        offset_x=args.proj_offset_x,
        offset_y=args.proj_offset_y,
        fullscreen=True,
    )

    # Black canvas with a white border rectangle spanning the full projector image
    proj = np.zeros((args.proj_h, args.proj_w, 3), np.uint8)
    cv2.rectangle(
        proj,
        (2, 2),
        (args.proj_w - 3, args.proj_h - 3),
        (255, 255, 255),
        2
    )
    cv2.imshow(proj_name, proj)
    cv2.waitKey(1)

    print("[calib] Adjust your projector / Lego board so this white rectangle")
    print("        roughly matches the Lego baseplate area.\n")

    # ---------- CAMERA SIDE ----------
    cap = cv2.VideoCapture(args.cam_index)
    if not cap.isOpened():
        cv2.destroyAllWindows()
        raise RuntimeError("Could not open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        cv2.destroyAllWindows()
        raise RuntimeError("Failed to read from camera for calibration")

    h_cam, w_cam = frame.shape[:2]
    print(f"[calib] CAMERA frame shape: {w_cam}x{h_cam} (W x H)")
    print("[calib] IMPORTANT: use the same cam_w/cam_h in your live CoTracker script,")
    print("        and do NOT manually resize the CAMERA window.\n")

    cam_name = "CAMERA"
    cv2.namedWindow(cam_name, cv2.WINDOW_AUTOSIZE)

    click_state = {"pts": []}
    cv2.setMouseCallback(cam_name, on_mouse, click_state)

    print("== Corner selection instructions ==")
    print("Look at the CAMERA window and click the 4 corners of the projected rectangle.")
    print("Order: top-left, top-right, bottom-right, bottom-left.")
    print("Press ESC to abort.\n")

    # Main loop: show camera, collect 4 clicks
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[calib] Camera read failed; exiting.")
            break

        # Draw already clicked points and their indices
        for idx, (cx, cy) in enumerate(click_state["pts"]):
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{idx+1}",
                        (int(cx)+5, int(cy)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, "Click rectangle corners: TL, TR, BR, BL",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(cam_name, frame)
        k = cv2.waitKey(10) & 0xFF

        if k == 27:  # ESC
            print("[calib] Aborted by user (ESC).")
            cap.release()
            cv2.destroyAllWindows()
            return

        if len(click_state["pts"]) == 4:
            break

    if len(click_state["pts"]) < 4:
        print("[calib] Not enough corners clicked; exiting.")
        cap.release()
        cv2.destroyAllWindows()
        return

    cam_pts = np.array(click_state["pts"], dtype=np.float32)
    print("[calib] CAMERA corners:\n", cam_pts)

    # Projector rectangle corners: full image
    proj_pts = np.array([
        [0, 0],                          # top-left
        [args.proj_w - 1, 0],            # top-right
        [args.proj_w - 1, args.proj_h - 1],  # bottom-right
        [0, args.proj_h - 1],            # bottom-left
    ], dtype=np.float32)
    print("[calib] PROJECTOR corners:\n", proj_pts)

    # Compute homography camera -> projector
    H, mask = cv2.findHomography(cam_pts, proj_pts, method=0)
    if H is None:
        print("[calib] ERROR: homography computation failed.")
    else:
        np.save(args.out, H)
        print(f"[calib] Saved homography (3x3) to {args.out}")
        print(H)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

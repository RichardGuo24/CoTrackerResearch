import cv2
import numpy as np
import argparse

# --------------------------- Mouse callback ---------------------------

def on_mouse_corner(event, x, y, flags, param):
    """
    Collect up to 4 Lego-board corners in CAMERA view.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        pts = param["pts"]
        if len(pts) < 4:
            pts.append((float(x), float(y)))
            print(f"[lego calib] corner #{len(pts)} at CAMERA ({x}, {y})")

def on_mouse_dot(event, x, y, flags, param):
    """
    Collect a single click for the current dot in CAMERA view.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        param["clicked"] = (float(x), float(y))
        print(f"[lego calib] dot click at CAMERA ({x}, {y})")


# --------------------------- Projector window ---------------------------

def make_projector_window(name, proj_w, proj_h, offset_x=0, offset_y=0, fullscreen=True):
    """
    Create and position the projector window on the projector display.
    offset_x/offset_y place the window on the correct physical screen.
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, offset_x, offset_y)
    cv2.resizeWindow(name, proj_w, proj_h)
    if fullscreen:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_index", type=int, default=0,
                    help="Camera index for Lego setup")
    ap.add_argument("--cam_w", type=int, default=1920,
                    help="Camera capture width (must match live script)")
    ap.add_argument("--cam_h", type=int, default=1080,
                    help="Camera capture height (must match live script)")
    ap.add_argument("--proj_w", type=int, default=1920,
                    help="Projector resolution width")
    ap.add_argument("--proj_h", type=int, default=1080,
                    help="Projector resolution height")
    ap.add_argument("--proj_offset_x", type=int, default=2560,
                    help="X offset so window appears on projector display")
    ap.add_argument("--proj_offset_y", type=int, default=0,
                    help="Y offset so window appears on projector display")
    ap.add_argument("--margin_frac", type=float, default=0.1,
                    help="Fractional margin around Lego region in projector space (0–0.5)")
    ap.add_argument("--rows", type=int, default=3,
                    help="Grid rows of refinement dots inside Lego region")
    ap.add_argument("--cols", type=int, default=3,
                    help="Grid cols of refinement dots inside Lego region")
    ap.add_argument("--out", type=str, default="homography_lego.npy",
                    help="Output .npy file for 3x3 Lego homography")
    args = ap.parse_args()

    # -------------------- Projector setup --------------------
    proj_name = "PROJECTOR"
    make_projector_window(
        proj_name,
        proj_w=args.proj_w,
        proj_h=args.proj_h,
        offset_x=args.proj_offset_x,
        offset_y=args.proj_offset_y,
        fullscreen=True
    )

    # Initially show a blank black canvas (no rectangle yet)
    proj_canvas = np.zeros((args.proj_h, args.proj_w, 3), np.uint8)
    cv2.imshow(proj_name, proj_canvas)
    cv2.waitKey(1)

    # -------------------- Camera setup --------------------
    cap = cv2.VideoCapture(args.cam_index,cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        cv2.destroyAllWindows()
        raise RuntimeError("Failed to read from camera")

    h_cam, w_cam = frame.shape[:2]
    print(f"[lego calib] CAMERA frame: {w_cam}x{h_cam} (W x H)")
    print("[lego calib] IMPORTANT: use the same cam_w/cam_h in your live CoTracker script.\n")

    cam_name = "CAMERA"
    cv2.namedWindow(cam_name, cv2.WINDOW_AUTOSIZE)

    # -------------------- Stage 1: click Lego corners --------------------
    print("[lego calib] Stage 1: Click the 4 LEGO BASEPLATE corners in CAMERA view.")
    print("             Order: top-left, top-right, bottom-right, bottom-left.")
    print("             Press ESC to abort.\n")

    corner_state = {"pts": []}
    cv2.setMouseCallback(cam_name, on_mouse_corner, corner_state)

    # Loop until we have 4 corners
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[lego calib] Camera read failed; exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return

        # Draw already clicked corners
        for idx, (cx, cy) in enumerate(corner_state["pts"]):
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{idx+1}",
                        (int(cx)+5, int(cy)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, "Click LEGO corners: TL, TR, BR, BL",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(cam_name, frame)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            print("[lego calib] Aborted by user (ESC) during corner selection.")
            cap.release()
            cv2.destroyAllWindows()
            return

        if len(corner_state["pts"]) == 4:
            break

    cam_lego_corners = np.array(corner_state["pts"], dtype=np.float32)
    print("[lego calib] LEGO corners in CAMERA coords:\n", cam_lego_corners)

    # -------------------- Define Lego region in PROJECTOR space --------------------
    Wp, Hp = args.proj_w, args.proj_h
    margin = max(0.0, min(args.margin_frac, 0.49))  # clamp to [0, 0.49]

    x_min = margin * Wp
    x_max = (1.0 - margin) * Wp
    y_min = margin * Hp
    y_max = (1.0 - margin) * Hp

    proj_lego_corners = np.array([
        [x_min, y_min],  # top-left
        [x_max, y_min],  # top-right
        [x_max, y_max],  # bottom-right
        [x_min, y_max],  # bottom-left
    ], dtype=np.float32)

    print("[lego calib] LEGO region in PROJECTOR coords (canonical rectangle):\n", proj_lego_corners)

    # Initial homography from LEGO corners alone
    H0, mask0 = cv2.findHomography(cam_lego_corners, proj_lego_corners, method=0)
    if H0 is None:
        print("[lego calib] ERROR: could not compute initial homography from corners.")
        cap.release()
        cv2.destroyAllWindows()
        return
    print("[lego calib] Initial homography H0 (from 4 Lego corners):\n", H0)

    # Optionally: draw the Lego rectangle on projector for visualization
    proj_canvas = np.zeros((Hp, Wp, 3), np.uint8)
    # Draw rectangle edges
    for i in range(4):
        p1 = tuple(proj_lego_corners[i].astype(int))
        p2 = tuple(proj_lego_corners[(i+1) % 4].astype(int))
        cv2.line(proj_canvas, p1, p2, (0, 255, 0), 2)
    cv2.imshow(proj_name, proj_canvas)
    cv2.waitKey(500)  # brief pause so you can see it

    # -------------------- Stage 2: refinement dots INSIDE Lego region --------------------
    print("\n[lego calib] Stage 2: Refinement with grid of dots inside LEGO region.")
    print("             For each dot: look at CAMERA and click on the dot.")
    print("             Press ESC to abort.\n")

    # Build grid of projector points inside the Lego rectangle
    rows = max(1, args.rows)
    cols = max(1, args.cols)
    xs = np.linspace(x_min, x_max, cols)
    ys = np.linspace(y_min, y_max, rows)

    proj_grid_pts = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            proj_grid_pts.append([x, y])
    proj_grid_pts = np.array(proj_grid_pts, dtype=np.float32)

    cam_grid_pts = []

    dot_state = {"clicked": None}
    cv2.setMouseCallback(cam_name, on_mouse_dot, dot_state)

    try:
        for idx, (u, v) in enumerate(proj_grid_pts):
            print(f"[lego calib] Dot {idx+1}/{len(proj_grid_pts)} at PROJECTOR ({u:.1f}, {v:.1f})")

            # Draw this dot over the Lego region outline
            proj_canvas = np.zeros((Hp, Wp, 3), np.uint8)
            # Lego rectangle outline
            for i in range(4):
                p1 = tuple(proj_lego_corners[i].astype(int))
                p2 = tuple(proj_lego_corners[(i+1) % 4].astype(int))
                cv2.line(proj_canvas, p1, p2, (0, 255, 0), 2)
            cv2.circle(proj_canvas, (int(u), int(v)), 10, (255, 255, 255), -1)
            cv2.imshow(proj_name, proj_canvas)
            cv2.waitKey(1)

            dot_state["clicked"] = None

            # Wait for user to click this dot in CAMERA
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[lego calib] Camera read failed during dot stage; exiting.")
                    return

                # Show previously clicked refinement points for feedback
                for (cx, cy) in cam_grid_pts:
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)

                cv2.putText(frame, f"Click dot {idx+1}/{len(proj_grid_pts)}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(cam_name, frame)
                k = cv2.waitKey(10) & 0xFF
                if k == 27:
                    print("[lego calib] Aborted by user (ESC) during dot refinement.")
                    return

                if dot_state["clicked"] is not None:
                    cam_grid_pts.append(dot_state["clicked"])
                    break

        cam_grid_pts = np.array(cam_grid_pts, dtype=np.float32)
        print("[lego calib] Refinement CAMERA points:\n", cam_grid_pts)
        print("[lego calib] Refinement PROJECTOR points:\n", proj_grid_pts)

        # -------------------- Final homography with all points --------------------
        cam_all = np.vstack([cam_lego_corners, cam_grid_pts])
        proj_all = np.vstack([proj_lego_corners, proj_grid_pts])

        print("[lego calib] Computing final homography H (camera -> projector) with RANSAC...")
        H_final, mask = cv2.findHomography(cam_all, proj_all, cv2.RANSAC)
        if H_final is None:
            print("[lego calib] ERROR: final homography computation failed.")
            return

        np.save(args.out, H_final)
        print(f"[lego calib] Saved final LEGO homography (3x3) to {args.out}")
        print(H_final)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

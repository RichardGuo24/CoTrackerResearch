import cv2, numpy as np
import argparse
from utils.display import make_fullscreen_window

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts = param["pts"]
        if len(pts) < 4:
            pts.append((x, y))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_index", type=int, default=0)
    ap.add_argument("--proj_w", type=int, default=1280)
    ap.add_argument("--proj_h", type=int, default=720)
    ap.add_argument("--out", default="homography.npy")
    args = ap.parse_args()

    # Projector canvas (all black with thin white border)
    proj = np.zeros((args.proj_h, args.proj_w, 3), np.uint8)
    cv2.rectangle(proj, (2,2), (args.proj_w-3, args.proj_h-3), (255,255,255), 2)
    make_fullscreen_window("PROJECTOR", x=0, y=0, fullscreen=True)
    cv2.imshow("PROJECTOR", proj)
    cv2.waitKey(1)

    cap = cv2.VideoCapture(args.cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    click_state = {"pts": []}
    cv2.namedWindow("CAMERA")
    cv2.setMouseCallback("CAMERA", on_mouse, click_state)

    print("Click the 4 projector rectangle corners in CAMERA view, clockwise from top-left.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Draw already clicked points
        for (x,y) in click_state["pts"]:
            cv2.circle(frame, (x,y), 5, (0,255,0), -1)

        cv2.imshow("CAMERA", frame)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:  # ESC to abort
            cap.release()
            cv2.destroyAllWindows()
            return
        if len(click_state["pts"]) == 4:
            cam_pts = np.array(click_state["pts"], dtype=np.float32)
            proj_pts = np.array([[0,0],
                                 [args.proj_w-1, 0],
                                 [args.proj_w-1, args.proj_h-1],
                                 [0, args.proj_h-1]], dtype=np.float32)
            H, _ = cv2.findHomography(cam_pts, proj_pts, method=0)
            np.save(args.out, H)
            print(f"Saved homography to {args.out}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

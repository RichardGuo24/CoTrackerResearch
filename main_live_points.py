import argparse
import time
import numpy as np
import cv2
import torch

# --------------------------- Helpers ---------------------------

def best_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def resize_keep_aspect(img_rgb, max_side):
    h, w = img_rgb.shape[:2]
    if max(h, w) <= max_side:
        return img_rgb, 1.0, 1.0
    if h >= w:
        scale = max_side / float(h)
    else:
        scale = max_side / float(w)
    new_w, new_h = int(round(w*scale)), int(round(h*scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    sx = w / float(new_w)  # resized -> original: multiply x by sx
    sy = h / float(new_h)  # resized -> original: multiply y by sy
    return resized, sx, sy

def to_video_chunk(frames_rgb_uint8, device):
    arr = np.stack(frames_rgb_uint8, axis=0)              # (T,H,W,3) uint8
    ten = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    return ten.permute(0, 3, 1, 2).unsqueeze(0).contiguous()  # (1,T,3,H,W)

def transform_points_homography(points_xy, H, proj_w, proj_h):
    if H is None or points_xy is None or len(points_xy) == 0:
        return None
    pts = np.concatenate([points_xy, np.ones((points_xy.shape[0], 1))], axis=1)  # (N,3)
    mapped = (H @ pts.T).T
    w = mapped[:, 2:3]
    w[w == 0] = 1e-9
    proj_xy = mapped[:, :2] / w
    proj_xy[:, 0] = np.clip(proj_xy[:, 0], 0, proj_w - 1)
    proj_xy[:, 1] = np.clip(proj_xy[:, 1], 0, proj_h - 1)
    return proj_xy

# --------------------------- Mouse for point picking ---------------------------

class ClickBuffer:
    def __init__(self):
        self.points_cam = []  # clicked points in CAMERA coordinates (original camera resolution)

    def add(self, x, y):
        self.points_cam.append((float(x), float(y)))

    def clear(self):
        self.points_cam = []

def make_mouse_callback(clickbuf):
    def _cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clickbuf.add(x, y)
    return _cb

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_index", type=int, default=0)
    ap.add_argument("--cam_w", type=int, default=640)
    ap.add_argument("--cam_h", type=int, default=480)
    ap.add_argument("--proj_w", type=int, default=1280)
    ap.add_argument("--proj_h", type=int, default=720)
    ap.add_argument("--grid_size", type=int, default=10)
    ap.add_argument("--model_max_side", type=int, default=512)
    ap.add_argument("--homography", default="homography.npy")
    args = ap.parse_args()

    device = best_device()
    print(f"[info] device={device}")

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
    step = int(getattr(model, "step", 8))
    win  = 2 * step
    print(f"[info] model.step={step}, window={win}")

    # Load homography if present
    H = None
    try:
        H_loaded = np.load(args.homography)
        if isinstance(H_loaded, np.ndarray) and H_loaded.shape == (3, 3):
            H = H_loaded
            print(f"[info] loaded homography from {args.homography}")
        else:
            print(f"[warn] {args.homography} is not a 3x3 matrix; ignoring.")
    except Exception as e:
        print(f"[warn] homography load failed or not provided: {e}")

    # Open camera
    cap = cv2.VideoCapture(args.cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    # Windows
    proj_win = "PROJECTOR"
    cam_win  = "CAMERA"
    cv2.namedWindow(proj_win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(proj_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow(cam_win, cv2.WINDOW_NORMAL)

    # Mouse for picking points
    clicks = ClickBuffer()
    cv2.setMouseCallback(cam_win, make_mouse_callback(clicks))

    # Buffers/state
    buf_model_rgb = []
    scale_x, scale_y = 1.0, 1.0
    frames_seen = 0
    is_first = True
    seed_mode = "grid"   # "grid" or "manual"
    need_reseed = True   # seed at first opportunity

    last_xy = None
    last_vis = None
    t0 = time.time()

    print("[hint] Click points in CAMERA window. Press 'R' to (re)seed with clicked points, 'G' for grid, 'X' to clear, ESC to quit.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[warn] camera read failed; exiting.")
                break

            frames_seen += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            resized, sx, sy = resize_keep_aspect(frame_rgb, args.model_max_side)
            scale_x, scale_y = sx, sy

            buf_model_rgb.append(resized)
            if len(buf_model_rgb) > win:
                buf_model_rgb = buf_model_rgb[-win:]

            # Decide if we should attempt a model step
            do_step = (len(buf_model_rgb) == win) and (frames_seen % step == 0)

            if do_step:
                video_chunk = to_video_chunk(buf_model_rgb, device)

                # If user requested reseed, we set is_first=True and prepare query_points if manual
                query_points_tensor = None
                if need_reseed:
                    is_first = True
                    if seed_mode == "manual":
                        # Map clicked CAM points -> resized coords expected by model
                        if len(clicks.points_cam) == 0:
                            print("[warn] no clicked points; falling back to grid seeding.")
                        else:
                            pts = []
                            for (xc, yc) in clicks.points_cam:
                                xr = xc / scale_x
                                yr = yc / scale_y
                                pts.append([xr, yr])  # (x,y) in resized pixel coords
                            pts = np.array(pts, dtype=np.float32)  # (N,2)
                            # CoTracker online expects query_points as (B, 1, N, 2) in pixel coords
                            query_points_tensor = torch.from_numpy(pts).to(device=device, dtype=torch.float32)
                            query_points_tensor = query_points_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,N,2)
                    need_reseed = False

                # Call model with robust unpacking
                with torch.inference_mode():
                    if is_first:
                        if seed_mode == "manual" and query_points_tensor is not None:
                            out = model(
                                video_chunk,
                                is_first_step=True,
                                query_points=query_points_tensor,
                                grid_size=None,
                                grid_query_frame=0,
                            )
                        else:
                            out = model(
                                video_chunk,
                                is_first_step=True,
                                grid_size=args.grid_size,
                                grid_query_frame=0,
                            )
                        is_first = False
                    else:
                        out = model(video_chunk)

                pred_tracks = pred_vis = None
                if out is None:
                    print("[warn] model returned None on this step; waiting for next window…")
                elif isinstance(out, tuple) and len(out) == 2:
                    pred_tracks, pred_vis = out
                elif isinstance(out, dict):
                    pred_tracks = out.get("pred_tracks") or out.get("tracks")
                    pred_vis    = out.get("pred_visibility") or out.get("visibility")
                else:
                    print(f"[warn] unexpected model output type: {type(out)}; skipping this step.")

                if pred_tracks is not None and pred_vis is not None:
                    try:
                        last_xy  = pred_tracks[0, -1].detach().cpu().numpy()  # (N,2) in resized coords
                        last_vis = pred_vis[0, -1].detach().cpu().numpy()     # (N,)
                    except Exception as e:
                        print(f"[warn] could not index tracks/vis: {e}")
                        last_xy = None
                        last_vis = None

            # --------- Render ---------
            preview = frame_rgb.copy()

            # Draw selected (clicked) points for user feedback
            for (x, y) in clicks.points_cam:
                cv2.circle(preview, (int(x), int(y)), 4, (0, 200, 255), -1)  # orange for selected

            # Draw tracked points (green)
            if last_xy is not None and last_vis is not None:
                cam_xy = last_xy.copy()
                cam_xy[:, 0] *= scale_x
                cam_xy[:, 1] *= scale_y
                for (x, y), v in zip(cam_xy, last_vis):
                    if v > 0.5:
                        cv2.circle(preview, (int(x), int(y)), 2, (0, 255, 0), -1)

            cv2.imshow(cam_win, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

            # Projector canvas
            proj_canvas = np.zeros((args.proj_h, args.proj_w, 3), np.uint8)
            if H is not None and last_xy is not None and last_vis is not None:
                cam_xy = last_xy.copy()
                cam_xy[:, 0] *= scale_x
                cam_xy[:, 1] *= scale_y
                proj_xy = transform_points_homography(cam_xy, H, args.proj_w, args.proj_h)
                if proj_xy is not None:
                    for (x, y), v in zip(proj_xy, last_vis):
                        if v > 0.5:
                            cv2.circle(proj_canvas, (int(x), int(y)), 4, (255, 255, 255), -1)
            cv2.imshow(proj_win, proj_canvas)

            # Keys
            k = cv2.waitKey(1) & 0xFF
            if k == 27:   # ESC
                break
            elif k in (ord('x'), ord('X')):
                clicks.clear()
                print("[info] cleared clicked points.")
            elif k in (ord('r'), ord('R')):
                seed_mode = "manual"
                need_reseed = True
                print("[info] will reseed with clicked points on next window.")
            elif k in (ord('g'), ord('G')):
                seed_mode = "grid"
                need_reseed = True
                print("[info] will reseed with grid on next window.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - t0
        fps = (frames_seen / elapsed) if elapsed > 0 else 0.0
        print(f"[info] elapsed={elapsed:.2f}s frames={frames_seen} avg_fps={fps:.2f}")

if __name__ == "__main__":
    main()

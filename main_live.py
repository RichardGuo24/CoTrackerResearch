# main_live.py
# Live camera -> CoTracker (online) -> optional homography -> projector overlay
# Self-contained, defensive against first-step/shape/return-type pitfalls.

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
    """Return (resized_rgb, scale_x, scale_y) where scale_x, scale_y map resized->original."""
    h, w = img_rgb.shape[:2]
    if max(h, w) <= max_side:
        return img_rgb, 1.0, 1.0
    if h >= w:
        scale = max_side / float(h)
    else:
        scale = max_side / float(w)
    new_w, new_h = int(round(w*scale)), int(round(h*scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # scale-back factors (multiply resized coords by these to get original camera coords)
    sx = w / float(new_w)
    sy = h / float(new_h)
    return resized, sx, sy

def to_video_chunk(frames_rgb_uint8, device):
    """frames: list of (H,W,3) uint8 RGB -> torch float32 (1,T,3,H,W) on device, values 0..255."""
    arr = np.stack(frames_rgb_uint8, axis=0)              # (T,H,W,3) uint8
    ten = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    return ten.permute(0, 3, 1, 2).unsqueeze(0).contiguous()  # (1,T,3,H,W)

def transform_points_homography(points_xy, H, proj_w, proj_h):
    """(N,2) camera pixels -> (N,2) projector pixels using 3x3 H. Clamped to projector frame."""
    if H is None or points_xy is None or len(points_xy) == 0:
        return None
    pts = np.concatenate([points_xy, np.ones((points_xy.shape[0], 1))], axis=1)  # (N,3)
    mapped = (H @ pts.T).T
    w = mapped[:, 2:3]
    # Avoid divide-by-zero
    w[w == 0] = 1e-9
    proj_xy = mapped[:, :2] / w
    proj_xy[:, 0] = np.clip(proj_xy[:, 0], 0, proj_w - 1)
    proj_xy[:, 1] = np.clip(proj_xy[:, 1], 0, proj_h - 1)
    return proj_xy

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_index", type=int, default=0)
    ap.add_argument("--cam_w", type=int, default=640)
    ap.add_argument("--cam_h", type=int, default=480)
    ap.add_argument("--proj_w", type=int, default=1280)
    ap.add_argument("--proj_h", type=int, default=720)
    ap.add_argument("--grid_size", type=int, default=10)
    ap.add_argument("--model_max_side", type=int, default=512)  # model input size cap
    ap.add_argument("--homography", default="homography.npy", help="3x3 camera->projector npy file (optional)")
    args = ap.parse_args()

    device = best_device()
    print(f"[info] device={device}")

    # Load model (online/streaming variant)
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
    # CoTracker online expects windows of length = 2*step and you should call it every `step` frames
    step = int(getattr(model, "step", 8))
    win  = 2 * step
    print(f"[info] model.step={step}, window={win}")

    # Load homography if provided
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

    # Windows: one fullscreen projector canvas, one camera preview
    proj_win = "PROJECTOR"
    cam_win  = "CAMERA"
    cv2.namedWindow(proj_win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(proj_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow(cam_win, cv2.WINDOW_NORMAL)

    # Ring buffers (lists with manual trimming, no extra deps)
    buf_model_rgb = []  # resized frames for model (fast)
    scale_x, scale_y = 1.0, 1.0  # resized->camera scaling
    frames_seen = 0
    is_first = True

    t0 = time.time()
    last_xy = None
    last_vis = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[warn] camera read failed; exiting.")
                break

            frames_seen += 1

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Prepare resized copy for model & remember scale-back to camera coords
            resized, sx, sy = resize_keep_aspect(frame_rgb, args.model_max_side)
            scale_x, scale_y = sx, sy

            # Buffer for model
            buf_model_rgb.append(resized)
            if len(buf_model_rgb) > win:
                buf_model_rgb = buf_model_rgb[-win:]  # keep last window

            # When we have exactly a full window and hit a step boundary, call model
            if len(buf_model_rgb) == win and (frames_seen % step == 0):
                # Build tensor (1,T,3,H,W) float32 0..255
                video_chunk = to_video_chunk(buf_model_rgb, device)

                # Defensive call: different versions may return tuple/dict/None
                with torch.inference_mode():
                    if is_first:
                        out = model(
                            video_chunk,
                            is_first_step=True,
                            grid_size=args.grid_size,
                            grid_query_frame=0,  # seed at the first frame of the chunk
                        )
                        is_first = False
                    else:
                        out = model(video_chunk)

                pred_tracks = pred_vis = None
                if out is None:
                    # Some builds return nothing on first call; just wait for the next step
                    print("[warn] model returned None on this step (likely first call); continuing.")
                elif isinstance(out, tuple):
                    if len(out) == 2:
                        pred_tracks, pred_vis = out
                elif isinstance(out, dict):
                    pred_tracks = out.get("pred_tracks") or out.get("tracks")
                    pred_vis    = out.get("pred_visibility") or out.get("visibility")
                else:
                    # Unexpected type — print once, then continue safely
                    print(f"[warn] unexpected model output type: {type(out)}; skipping this step.")

                # Validate tensors before indexing
                if pred_tracks is not None and pred_vis is not None:
                    try:
                        # Expect (B,T,N,2) and (B,T,N)
                        last_xy  = pred_tracks[0, -1].detach().cpu().numpy()  # (N,2) in resized coords
                        last_vis = pred_vis[0, -1].detach().cpu().numpy()     # (N,)
                    except Exception as e:
                        print(f"[warn] could not index tracks/vis: {e}")
                        last_xy = None
                        last_vis = None

            # ---------------- Render ----------------
            # Camera preview: draw green dots (for human sanity-check)
            preview = frame_rgb.copy()
            if last_xy is not None and last_vis is not None:
                # Map from resized coords back to camera coords
                cam_xy = last_xy.copy()
                cam_xy[:, 0] *= scale_x
                cam_xy[:, 1] *= scale_y
                for (x, y), v in zip(cam_xy, last_vis):
                    if v > 0.5:
                        cv2.circle(preview, (int(x), int(y)), 2, (0, 255, 0), -1)

            cv2.imshow(cam_win, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

            # Projector: black canvas + white dots at projector pixels (if H is available)
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

            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - t0
        fps = frames_seen / elapsed if elapsed > 0 else 0.0
        print(f"[info] elapsed={elapsed:.2f}s frames={frames_seen} avg_fps={fps:.2f}")

if __name__ == "__main__":
    main()

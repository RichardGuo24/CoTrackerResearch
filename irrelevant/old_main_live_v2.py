import argparse
import time
import numpy as np
import cv2
import torch

# ================== CONFIG KNOBS (TUNE HERE) ==================

VIS_THRESH = 0.5          # visibility threshold for points
OUTLIER_SCALE = 2.5       # how aggressively to drop outlier points in cluster
COLOR_PATCH_RADIUS = 10   # radius (pixels) of color template patch
COLOR_DIST_THRESH = 40.0  # threshold for color mismatch (0..255 RGB distance)
SMOOTH_ALPHA = 0.7        # how strongly to keep previous center vs new (0..1)

# ===============================================================


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
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    sx = w / float(new_w)  # multiply resized x by sx -> camera x
    sy = h / float(new_h)  # multiply resized y by sy -> camera y
    return resized, sx, sy


def to_video_chunk(frames_rgb_uint8, device):
    """frames: list of (H,W,3) uint8 RGB -> torch float32 (1,T,3,H,W) on device, values 0..255."""
    arr = np.stack(frames_rgb_uint8, axis=0)  # (T,H,W,3)
    ten = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    return ten.permute(0, 3, 1, 2).unsqueeze(0).contiguous()  # (1,T,3,H,W)


def transform_points_homography(points_xy, H, proj_w, proj_h):
    """(N,2) camera pixels -> (N,2) projector pixels using 3x3 H. Clamped to projector frame."""
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


# --------------------------- Utility: cluster & color helpers ---------------------------

def robust_cluster_center(cam_xy, vis):
    """
    Compute a robust center of a set of points in camera space:
    - filter by visibility
    - remove distance outliers
    - return (center, filtered_points, filtered_vis)
    """
    if cam_xy is None or vis is None or len(cam_xy) == 0:
        return None, None, None

    vis = np.asarray(vis)
    cam_xy = np.asarray(cam_xy)

    mask = vis > VIS_THRESH
    if not np.any(mask):
        return None, None, None

    pts = cam_xy[mask]  # (M,2)
    if pts.shape[0] == 0:
        return None, None, None

    # Initial center
    center = pts.mean(axis=0)

    # Remove outliers based on distance to center
    dists = np.linalg.norm(pts - center[None, :], axis=1)
    if pts.shape[0] > 2:
        med = np.median(dists)
        # keep points not too far from median distance
        keep = dists < max(5.0, OUTLIER_SCALE * med)
        pts = pts[keep]
        vis_filtered = np.ones(len(pts), dtype=np.float32)
    else:
        vis_filtered = np.ones(len(pts), dtype=np.float32)

    if pts.shape[0] == 0:
        return None, None, None

    center = pts.mean(axis=0)
    return center, pts, vis_filtered


def extract_patch_mean_color(frame_rgb, center_xy, radius):
    """Extract mean RGB color around center (in camera coords)."""
    if center_xy is None:
        return None
    h, w, _ = frame_rgb.shape
    cx, cy = int(center_xy[0]), int(center_xy[1])
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame_rgb[y0:y1, x0:x1]  # (H, W, 3)
    if patch.size == 0:
        return None
    return patch.reshape(-1, 3).mean(axis=0)  # (3,)


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

    # Load CoTracker online model
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
    step = int(getattr(model, "step", 8))
    win = 2 * step
    print(f"[info] model.step={step}, window={win}")

    # Load homography if present
    H = None    # camera->projector
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
    cap = cv2.VideoCapture(args.cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    # Windows
    proj_win = "PROJECTOR"
    cam_win = "CAMERA"
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
    need_reseed = True   # force seed on first call
    queries_tensor = None

    # Tracking result state
    last_xy = None
    last_vis = None

    # Cluster & smoothing state
    prev_center_cam = None
    curr_center_cam = None
    interp_center_cam = None
    frames_since_update = 0
    frames_per_step = step
    template_color = None  # appearance model for manual mode

    t0 = time.time()
    print("[hint] Click points in CAMERA window. Press 'R' to reseed with clicked points, 'G' for grid, 'X' to clear, ESC to quit.")

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

            do_step = (len(buf_model_rgb) == win) and (frames_seen % step == 0)

            if do_step:
                video_chunk = to_video_chunk(buf_model_rgb, device)

                # Prepare queries if we requested a reseed
                if need_reseed:
                    is_first = True
                    queries_tensor = None
                    if seed_mode == "manual":
                        if len(clicks.points_cam) == 0:
                            print("[warn] no clicked points; falling back to grid seeding.")
                        else:
                            # Map camera clicks -> resized coords -> queries (t, x, y)
                            pts = []
                            for (xc, yc) in clicks.points_cam:
                                xr = xc / scale_x
                                yr = yc / scale_y
                                pts.append([0.0, xr, yr])  # t=0
                            pts = np.array(pts, dtype=np.float32)  # (N,3)
                            queries_tensor = torch.from_numpy(pts).to(device=device, dtype=torch.float32)
                            queries_tensor = queries_tensor.unsqueeze(0)  # (1,N,3)
                            print(f"[info] manual reseed with {pts.shape[0]} points.")
                    need_reseed = False

                # Call model with robust unpacking
                with torch.inference_mode():
                    if is_first:
                        if seed_mode == "manual" and queries_tensor is not None:
                            out = model(
                                video_chunk,
                                is_first_step=True,
                                queries=queries_tensor,
                                grid_size=0,
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
                        if seed_mode == "manual" and queries_tensor is not None:
                            out = model(video_chunk, queries=queries_tensor)
                        else:
                            out = model(video_chunk)

                pred_tracks = pred_vis = None
                if out is None:
                    print("[warn] model returned None on this step; waiting for next window…")
                elif isinstance(out, tuple) and len(out) == 2:
                    pred_tracks, pred_vis = out
                elif isinstance(out, dict):
                    pred_tracks = out.get("pred_tracks") or out.get("tracks")
                    pred_vis = out.get("pred_visibility") or out.get("visibility")
                else:
                    print(f"[warn] unexpected model output type: {type(out)}; skipping this step.")

                if pred_tracks is not None and pred_vis is not None:
                    try:
                        # (B,T,N,2) -> (N,2) for last frame
                        last_xy = pred_tracks[0, -1].detach().cpu().numpy()
                        last_vis = pred_vis[0, -1].detach().cpu().numpy()
                    except Exception as e:
                        print(f"[warn] could not index tracks/vis: {e}")
                        last_xy = None
                        last_vis = None

                    # --- NEW: compute robust cluster center in camera coords, with appearance sanity ---
                    if last_xy is not None and last_vis is not None:
                        cam_xy = last_xy.copy()
                        cam_xy[:, 0] *= scale_x
                        cam_xy[:, 1] *= scale_y

                        center_cam, pts_filtered, vis_filtered = robust_cluster_center(cam_xy, last_vis)

                        if center_cam is not None:
                            # Appearance check: compare mean color to template (only for manual mode)
                            patch_color = extract_patch_mean_color(frame_rgb, center_cam, COLOR_PATCH_RADIUS)
                            if seed_mode == "manual":
                                if template_color is None and patch_color is not None:
                                    # Initialize template on first successful track
                                    template_color = patch_color
                                    print("[info] initialized appearance template.")
                                elif patch_color is not None and template_color is not None:
                                    dist = np.linalg.norm(patch_color - template_color)
                                    if dist > COLOR_DIST_THRESH:
                                        # Large color mismatch: blend towards previous center to avoid big jumps
                                        print(f"[warn] color mismatch ({dist:.1f}); damping center update.")
                                        if prev_center_cam is not None:
                                            center_cam = SMOOTH_ALPHA * prev_center_cam + (1.0 - SMOOTH_ALPHA) * center_cam

                            # Temporal smoothing between updates (for center)
                            if curr_center_cam is None:
                                prev_center_cam = center_cam.copy()
                                curr_center_cam = center_cam.copy()
                            else:
                                prev_center_cam = curr_center_cam.copy()
                                curr_center_cam = center_cam.copy()

                            frames_since_update = 0
                            frames_per_step = step

            # --------- Render (runs every frame) ---------

            # Temporal interpolation of the center for smoother motion
            if curr_center_cam is not None and prev_center_cam is not None:
                alpha = min(1.0, frames_since_update / max(1, frames_per_step))
                interp_center_cam = (1.0 - alpha) * prev_center_cam + alpha * curr_center_cam
                frames_since_update += 1

            # Build camera preview image
            preview = frame_rgb.copy()

            # Draw selected (clicked) points (orange)
            for (x, y) in clicks.points_cam:
                cv2.circle(preview, (int(x), int(y)), 4, (0, 200, 255), -1)

            # Draw tracked points (green) if available
            if last_xy is not None and last_vis is not None:
                cam_xy = last_xy.copy()
                cam_xy[:, 0] *= scale_x
                cam_xy[:, 1] *= scale_y
                for (x, y), v in zip(cam_xy, last_vis):
                    if v > VIS_THRESH:
                        cv2.circle(preview, (int(x), int(y)), 2, (0, 255, 0), -1)

            # Draw robust, smoothed center (blue)
            if interp_center_cam is not None:
                cx, cy = int(interp_center_cam[0]), int(interp_center_cam[1])
                cv2.circle(preview, (cx, cy), 6, (0, 0, 255), 2)

            cv2.imshow(cam_win, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

            # Projector canvas: only draw the smoothed center (clean AR target)
            proj_canvas = np.zeros((args.proj_h, args.proj_w, 3), np.uint8)
            if H is not None and interp_center_cam is not None:
                proj_xy = transform_points_homography(
                    np.array([interp_center_cam], dtype=np.float32), H, args.proj_w, args.proj_h
                )
                if proj_xy is not None:
                    px, py = int(proj_xy[0, 0]), int(proj_xy[0, 1])
                    cv2.circle(proj_canvas, (px, py), 8, (255, 255, 255), -1)

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
                template_color = None  # reset appearance template on reseed
                print("[info] will reseed with clicked points on next window.")
            elif k in (ord('g'), ord('G')):
                seed_mode = "grid"
                need_reseed = True
                template_color = None
                print("[info] will reseed with grid on next window.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - t0
        fps = (frames_seen / elapsed) if elapsed > 0 else 0.0
        print(f"[info] elapsed={elapsed:.2f}s frames={frames_seen} avg_fps={fps:.2f}")


if __name__ == "__main__":
    main()

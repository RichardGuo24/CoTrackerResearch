import cv2, time

DEVICE = 0
cap = cv2.VideoCapture(DEVICE, cv2.CAP_DSHOW)  # try DirectShow on Windows

# Force MJPG (needed for 4K on these UVC cams)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Ask for 4K; try 16MP still res next; then 1080p as fallback
for (w, h) in [ (1920, 1080)]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    time.sleep(0.1)  # let backend settle
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (got_w, got_h) == (w, h):
        break

if not cap.isOpened():
    print("❌ Cannot open camera"); exit()

fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc_str = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])
fps = cap.get(cv2.CAP_PROP_FPS)  # may be 0 on some drivers
print(f"FOURCC: {fourcc_str}  Resolution: {got_w}x{got_h}  FPS(reported): {fps:.1f}")
print("Press ESC or 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("❌ Failed to grab frame"); break
    cv2.imshow("Arducam Preview", frame)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')): break

cap.release()
cv2.destroyAllWindows()

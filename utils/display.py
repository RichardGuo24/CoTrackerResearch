import cv2

def make_fullscreen_window(win_name: str, x: int = 0, y: int = 0, fullscreen: bool = True):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    # Position is best-effort; with fullscreen, some OSes ignore (but harmless).
    cv2.moveWindow(win_name, x, y)

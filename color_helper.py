import numpy as np
import cv2

def get_limits(color):
    # BGR colors from color_map
    if color == [0, 255, 255]:       # Yellow
        lower1 = np.array([20, 100, 100], dtype=np.uint8)
        upper1 = np.array([35, 255, 255], dtype=np.uint8)

    elif color == [0, 255, 0]:       # Green
        lower1 = np.array([35, 60, 60], dtype=np.uint8)
        upper1 = np.array([85, 255, 255], dtype=np.uint8)

    elif color == [255, 0, 0]:       # Blue
        lower1 = np.array([90, 60, 60], dtype=np.uint8)
        upper1 = np.array([135, 255, 255], dtype=np.uint8)

    elif color == [0, 0, 255]:       # Red
        # Red wraps around the HSV hue boundary.
        lower1 = np.array([0, 60, 60], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)

        lower2 = np.array([170, 60, 60], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)

        return lower1, upper1, lower2, upper2

    else:
        raise ValueError(f"Unsupported color: {color}")

    # Second range is unused for non-red colors.
    lower2 = np.array([0, 0, 0], dtype=np.uint8)
    upper2 = np.array([0, 0, 0], dtype=np.uint8)

    return lower1, upper1, lower2, upper2
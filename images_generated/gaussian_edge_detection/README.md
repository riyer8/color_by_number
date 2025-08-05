### Gaussian Edge Detection

```Python
def detect_object_boundaries(pixels: np.ndarray, edge_threshold: int = 50) -> np.ndarray:
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) # grayscaled image
    blurred = cv2.GaussianBlur(gray, (25, 25), 0) # reducing noise
    edges = cv2.Canny(blurred, edge_threshold, edge_threshold * 2)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    return edges > 0
```

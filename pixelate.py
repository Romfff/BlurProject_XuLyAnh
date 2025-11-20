import cv2

def pixelate_region(image, rect, blocks=10):
    x, y, w, h = rect
    sub = image[y:y+h, x:x+w]
    small = cv2.resize(sub, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = image.copy()
    out[y:y+h, x:x+w] = pixelated
    return out

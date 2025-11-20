import cv2

def gaussian_blur_region(image, rect, ksize=(51, 51)):
    x, y, w, h = rect
    sub = image[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(sub, ksize, 0)
    out = image.copy()
    out[y:y+h, x:x+w] = blurred
    return out

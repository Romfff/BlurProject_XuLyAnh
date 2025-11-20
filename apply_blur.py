from gaussian import gaussian_blur_region
from pixelate import pixelate_region
import cv2

def apply_blur_to_image(image, boxes, method="gaussian", gaussian_kernel=(51,51), pixel_blocks=10, draw_boxes=False):
    out = image.copy()
    for rect in boxes:
        if method == "gaussian":
            out = gaussian_blur_region(out, rect, ksize=gaussian_kernel)
        else:
            out = pixelate_region(out, rect, blocks=pixel_blocks)

        if draw_boxes:
            x, y, w, h = rect
            cv2.rectangle(out, (x, y), (x+w, y+h), (0,255,0), 2)

    return out

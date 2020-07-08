from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def resize_pil_image(image: Image.Image, coef: Optional[float] = None, size: Optional[Tuple[int, int]] = None)\
        -> Image.Image:
    if size is None:
        assert coef is not None
        (width, height) = (int(image.width * coef), int(image.height * coef))
    else:
        assert isinstance(size, (list, tuple))
        assert len(size) == 2
        (width, height) = size
    im_resized = image.resize((width, height))
    return im_resized


def rotate_pil_image(image: Image.Image, angle_degree: float) -> Image.Image:
    return image.rotate(angle=angle_degree, expand=True)


def cv2_rotate_image(image: np.ndarray, angle: float, scale: float = 1.0):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

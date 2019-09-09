"""
This module implements necessary routines for applying a halftoning procedure in images
"""

import numpy as np
import cv2


def halftoning(img, sweep_method=0):
    """
    halftoning function implements the main algorithm of halftoning to an input image.

    Keyword arguments:
    img -- the image to be halftoned
    sweep_method -- how the image is going to be sweeped (it affects how the error propagation happens)
        0 (default): image is sweeped line per line
        1: image is sweeped line per line, but at the end of a line it goes to the nearest neighbor in next line
    """
    if sweep_method == 0:
        print("Line per line. Easier to implement.")
    else:
        print("Damn. That's hard.")

    return img
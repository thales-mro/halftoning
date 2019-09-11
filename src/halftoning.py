"""
This module implements necessary routines for applying a halftoning procedure in images
"""

import time
import numpy as np


# Masks used for error propagation
MASKS = {"floyd-steinberg": np.array([[0, 0, 7/16], [3/16, 5/16, 1/16]]),
         "stevenson-arce": np.array([[0, 0, 0, 0, 0, 32/200, 0],
                                     [12/200, 0, 26/200, 0, 30/200, 0, 16/200],
                                     [0, 12/200, 0, 26/200, 0, 12/200, 0],
                                     [5/200, 0, 12/200, 0, 12/200, 0, 5/200]]),
         "burkes": np.array([[0, 0, 0, 8/32, 4/32],
                             [2/32, 4/32, 8/32, 4/32, 2/32]]),
         "sierra": np.array([[0, 0, 0, 5/32, 3/32],
                             [2/32, 4/32, 5/32, 4/32, 2/32],
                             [0, 2/32, 3/32, 2/32, 0]]),
         "stucki": np.array([[0, 0, 0, 8/42, 4/42],
                             [2/42, 4/42, 8/42, 4/42, 2/42],
                             [1/42, 2/42, 4/42, 2/42, 1/42]]),
         "jarvis-judice-ninke": np.array([[0, 0, 0, 7/48, 5/48],
                                          [3/48, 5/48, 7/48, 5/48, 3/48],
                                          [1/48, 3/48, 5/48, 3/48, 1/48]])
}
def sweep(img, mask, sm):
    """
    It applies Floyd and Steinberg technique for error distribution
    The F-S mask is a 2x3 one, so appropriate padding has to be done 
    """
    start = time.time()
    result = np.zeros_like(img)
    mask_h, mask_w = mask.shape
    offset = mask_w//2
    img_padded = np.pad(img, ((0, mask_h - 1), (mask_w//2, mask_w//2)), 'constant')
    m = (mask, np.flip(mask, 1))
    direction = 1
    for j in range(img.shape[0]):
        if direction > 0:
            beginning = 0
            end = img.shape[1]
            mask_idx = 0
        else:
            beginning = img.shape[1] - 1
            end = 0
            mask_idx = 1
        for i in range(beginning, end, direction):
            if img_padded[j][(i + offset)] < 128:
                result[j][i] = 0
            else:
                result[j][i] = 1

            error = img_padded[j][(i + offset)] - result[j][i]*255
            img_padded[j:j+mask_h, i:i+mask_w] = (img_padded[j:j+mask_h, i:i+mask_w] + (error*m[mask_idx])).astype(np.uint8)
        direction *= sm
        
    result = result*255

    end = time.time()
    print(end - start)

    return result

def apply_halftoning(img, err_method="floyd-steinberg", sweep_method=1):
    """
    halftoning function implements the main algorithm of halftoning to an input image.

    Keyword arguments:
    img -- the image to be halftoned
    err_method -- the error distribution method to be applied
        floyd-steinberg: Floyd and Steinberg mask
        stevenson-arce: Stevenson and Arce mask
        burkes: Burkes mask
        sierra: Sierra mask
        stucki: Stucki mask
        jarvis-judice-ninke: Jarvis, Judice and Ninke mask
    sweep_method -- how the image is going to be sweeped
    (it affects how the error propagation happens)
        1 (default): image is sweeped from left to right, all lines
        -1: image is sweeped alternated, from left to right,
        then from right to left when line changes
    """


    if sweep_method == 0:
        print("Line per line. Easier to implement.")
    else:
        print("Damn. That's hard.")

    fs = sweep(img, MASKS[err_method], sweep_method)

    return fs
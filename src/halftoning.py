"""
This module implements necessary routines for applying a halftoning procedure in images
"""

import numpy as np
import cv2
import time

"""
Masks used for error propapation
    MASKS[0]: Floyd and Steinberg mask
    MASKS[1]: Stevenson and Arce mask
    MASKS[2]: Burkes mask
    MASKS[3]: Sierra mask
    MASKS[4]: Stucki mask
    MASKS[5]: Jarvis, Judice and Ninke mask
"""
MASKS = (np.array([[0, 0, 7/16], [3/16, 5/16, 1/16]]),
         np.array([[0, 0, 0, 0, 0, 32/200, 0],
                   [12/200, 0, 26/200, 0, 30/200, 0, 16/200],
                   [0, 12/200, 0, 26/200, 0, 12/200, 0],
                   [5/200, 0, 12/200, 0, 12/200, 0, 5/200]]),
         np.array([[0, 0, 0, 8/32, 4/32],
                   [2/32, 4/32, 8/32, 4/32, 2/32]]),
         np.array([[0, 0, 0, 5/32, 3/32],
                   [2/32, 4/32, 5/32, 4/32, 2/32],
                   [0, 2/32, 3/32, 2/32, 0]]),
         np.array([[0, 0, 0, 8/42, 4/42],
                   [2/42, 4/42, 8/42, 4/42, 2/42],
                   [1/42, 2/42, 4/42, 2/42, 1/42]]),
         np.array([[0, 0, 0, 7/48, 5/48],
                   [3/48, 5/48, 7/48, 5/48, 3/48],
                   [1/48, 3/48, 5/48, 3/48, 1/48]]))

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
        #print(beginning, end, direction)
        #print(direction)
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

def halftoning(img, err_method=0, sweep_method=1):
    """ 
    halftoning function implements the main algorithm of halftoning to an input image.

    Keyword arguments:
    img -- the image to be halftoned
    err_method -- the error distribution method to be applied
        0 (default): Floyd and Steinberg mask
        1: Stevenson and Arce mask
        2: Burkes mask
        3: Sierra mask
        4: Stucki mask
        5: Jarvis, Judice and Ninke mask
    sweep_method -- how the image is going to be sweeped
    (it affects how the error propagation happens)
        1 (default): image is sweeped line per line
        -1: image is sweeped line per line, 
        but at the end of a line it goes to the nearest neighbor in next line
    """


    if sweep_method == 0:
        print("Line per line. Easier to implement.")
    else:
        print("Damn. That's hard.")

    fs = sweep(img, MASKS[err_method], sweep_method)

    return fs
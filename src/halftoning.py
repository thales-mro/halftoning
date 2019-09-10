"""
This module implements necessary routines for applying a halftoning procedure in images
"""

import numpy as np
import cv2
import time

def sweep(img, mask):
    """
    It applies Floyd and Steinberg technique for error distribution
    The F-S mask is a 2x3 one, so appropriate padding has to be done  
    """
    start = time.time()
    result = np.zeros_like(img)
    #print(result.shape)
    #kernel = np.array([[0, 0, 7/16], [3/16, 5/16, 1/16]])
    mask_h, mask_w = mask.shape
    offset = mask_w//2
    print(mask.shape, offset)
    img_padded = np.pad(img, ((0, mask_h - 1), (mask_w//2, mask_w//2)), 'constant')
    # print(result.shape)

    for j in range(img.shape[0]):
        for i in range(img.shape[1]):

            #result[j][i] = 0 if img_padded[j][i + offset] < 128 else 0
            if img_padded[j][i + offset] < 128:
                result[j][i] = 0
            else:
                result[j][i] = 1

            error = img_padded[j][i + offset] - result[j][i]*255
            img_padded[j:j+mask_h, i:i+mask_w] += (mask*error).astype(np.uint8)
    result = result*255


    end = time.time()
    print(end - start)

    return result

    # for j in range(result.shape[0]):
    #     np.where(img_padded[j] < 128, print("Oi"), print("Tchau!"))
    #     result[j] = np.where(img[j] < 128, 0, 1)
    # result = result*255
    # print(result)
    # end = time.time()
    # print(end - start)

    # return result



def halftoning(img, err_method=0, sweep_method=0):
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
        0 (default): image is sweeped line per line
        1: image is sweeped line per line, 
        but at the end of a line it goes to the nearest neighbor in next line
    """
    masks = (np.array([[0, 0, 7/16], [3/16, 5/16, 1/16]]),
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

    if sweep_method == 0:
        print("Line per line. Easier to implement.")
    else:
        print("Damn. That's hard.")

    fs = sweep(img, masks[err_method])

    return fs
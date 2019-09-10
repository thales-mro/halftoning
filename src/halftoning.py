"""
This module implements necessary routines for applying a halftoning procedure in images
"""

import numpy as np
import cv2
import time

def floyd_steinberg(img):
    """
    It applies Floyd and Steinberg technique for error distribution
    The F-S mask is a 2x3 one, so appropriate padding has to be done  
    """
    start = time.time()
    result = np.zeros_like(img)
    #print(result.shape)
    kernel = np.array([[0, 0, 7/16], [3/16, 5/16, 1/16]])
    img_padded = np.pad(img, ((0, 1), (1, 1)), 'constant')
    print(result.shape)
    for j in range(result.shape[0]):
        for i in range(result.shape[1]):
            #np.where(img_padded[j][i + 1] < 128, 0, 1)
            if img_padded[j][i + 1] < 128:
                result[j][i] = 0
            else:
                result[j][i] = 1
            error = img_padded[j][i + 1] - result[j][i]*255
            #print(error)
            #print(img_padded[j:j+kernel.shape[0], i:i+kernel.shape[1]] )
            #print("Oi\n", (kernel*error).astype(np.uint8))
            img_padded[j:j+kernel.shape[0], i:i+kernel.shape[1]] += (kernel*error).astype(np.uint8)
            #print(img_padded[j:j+kernel.shape[0], i:i+kernel.shape[1]] )
            #img_padded[img_padded < 0] = 0
            #img_padded[img_padded > 255] = 255
            #print(img_padded)
    #result = np.where(img > 0, np.sum(img[-1:1, 0:1]), -1)
    result = result*255
    end = time.time()
    print(end - start)
    return result



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

    fs = floyd_steinberg(img)

    return fs
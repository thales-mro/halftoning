import cv2
import numpy as np
from halftoning import halftoning

def open_image(name, grayscale=0):
    """
    it makes calls for openCV functions for reading an image based on a name

    Keyword arguments:
    name -- the name of the image to be opened
    grayscale -- whether image is opened in grayscale or not
        0 (default): image opened normally (with all 3 color channels)
        1: image opened in grayscale form
    """
    img_name = 'input/' + name + '_colored' + '.png'
    if grayscale:
        return cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(img_name)

def save_image(name, image):
    """
    it makes calls for openCV function for saving an image based on a name (path)
    and the image itself

    Keyword arguments:
    name -- the name (path) of the image to be saved
    image -- the image itself (numpy array)
    """
    image_name = 'output/' + name + '.png'
    cv2.imwrite(image_name, image)

def main():
    """
    Entrypoint for the code of project 01 MO443/2s2019
    """

    images = [
        'baboon',
        #'monalisa',
        #'peppers',
        #'watch'
    ]
    arr = np.array([[250, 20, 120, 170, 0, 255, 0], [100, 200, 220, 15, 30, 50, 160]])

    for image_name in images:
        image = open_image(image_name)
        image_grayscale = open_image(image_name, 1)
        colored_ht = np.zeros_like(image)
        # colored_ht[:,:,0] = halftoning(image[:,:,0], 0, 1)
        # colored_ht[:,:,1] = halftoning(image[:,:,1], 0, 1)
        # colored_ht[:,:,2] = halftoning(image[:,:,2], 0, 1)
        ht = halftoning(image_grayscale, 2, -1)
        #halftoning(image[:, :, 0], 0)
        save_image(image_name, ht)
        #save_image(image_name + "_colored", colored_ht)

main()

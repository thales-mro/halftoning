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
        'baboon'#,
        #'monalisa',
        #'peppers',
        #'watch'
    ]
    arr = np.array([[1, 200, 189], [244, 5, 6]])

    for image_name in images:
        image = open_image(image_name)
        image_grayscale = open_image(image_name, 1)
        #save grayscale image
        ht = halftoning(image_grayscale, 0)
        #halftoning(image[:, :, 0], 0)
        save_image('test', ht)

main()

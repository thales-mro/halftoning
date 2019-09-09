import cv2
from halftoning import halftoning

def main():
    """
    Entrypoint for the code of project 01 MO443/2s2019
    """
    print("Hey!")
    images = [
        'input/baboon_colored.png',
        'input/monalisa_colored.png',
        'input/peppers_colored.png',
        'input/watch_colored.png'
    ]

    for image_name in images:
        image = cv2.imread(image_name)
        #print(image)
        image_grayscale = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        print(image_grayscale)
        #save grayscale image
        cv2.imwrite('output/test.png', image_grayscale)

main()

from PIL import Image
import numpy as np
from typing import Union, Tuple, List
from scipy.signal import convolve2d
from skimage import filters
import os
def generateIndexMap(gray_list: List[np.ndarray], w_size: int) -> np.ndarray:
    # Generate an index map for the refocusing application
    # Input:
    #   gray_list - List of K gray-scale images
    #   w_size - half window size used for focus measure computation
    # Output:
    #   index_map - mxn index map
    #               index_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)
    kernel = filters.laplace(np.zeros(3,3), dtype=np.float32)
    print(kernel)
    image_map = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
    '''
    k = w_size
    width = gray_list[0].shape[1]
    height = gray_list[0].shape[0]
    index_map = np.zeros((height, width))
    square_b = {}
    for i in range(height):
        for j in range(width):
            top = i-k if i-k >= 0 else 0
            down = i+k+1 if i+k+1 <= height else height
            left = j-k if j-k >= 0 else 0
            right = j+k+1 if j+k+1 <= width else width
            square_b[(i,j)] = [top,down,left,right]
            for index, img in enumerate(gray_list):
                square = np.full((2*k+1, 2*k+1), np.average(img[top:down,left:right]))
                #print(square.shape)
                square[top+k-i:down+k-i,left+k-j:right+k-j] = img[top:down,left:right]
                biggest = max(biggest, np.sum(np.gradient(np.gradient(square, axis=0), axis=0)) + np.sum(np.gradient(np.gradient(square, axis=1), axis=1)))
                biggest_idx = index
            index_map[i,j] = biggest_idx
    return index_map
    '''
    raise NotImplementedError


def loadFocalStack(focal_stack_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Load the focal stack
    # Input:
    #   focal_stack_dir - directory of the focal stack
    # Output:
    #   rgb_list - List of RGB images for varying focal lengths
    #   gray_list - List of gray-scale images for varying focal lengths
    images = os.listdir(focal_stack_dir)
    rgb_list = []
    gray_list = []
    for img in images:
        img_path = os.path.join(focal_stack_dir, img)
        rgb_img = Image.open(img_path)
        gray_img = rgb_img.convert('L')
        rgb_list.append(np.array(rgb_img))
        gray_list.append(np.array(gray_img))
    return rgb_list, gray_list
    raise NotImplementedError


def refocusApp(rgb_list: List[np.ndarray], depth_map: np.ndarray) -> None:
    # Refocusing application
    # Input:
    #   rgb_list - List of RGB images for varying focal lengths
    #   depth_map - mxn index map
    #               depth_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)
    raise NotImplementedError

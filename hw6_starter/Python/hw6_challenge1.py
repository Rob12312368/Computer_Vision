from PIL import Image
import numpy as np
from typing import Union, Tuple, List
import os
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import sys
def generateIndexMap(gray_list: List[np.ndarray], w_size: int) -> np.ndarray:
    # Generate an index map for the refocusing application
    # Input:
    #   gray_list - List of K gray-scale images
    #   w_size - half window size used for focus measure computation
    # Output:
    #   index_map - mxn index map
    #               index_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)
    mvag_size = 20
    mv_kernel = np.ones((mvag_size, mvag_size)) / (mvag_size * mvag_size)

    img_height, img_width = gray_list[0].shape
    lapla_values = np.zeros((img_height, img_width, len(gray_list)))
    laplacian_x = np.array([[1, -2, 1]])
    laplacian_y = np.array([[1],[-2],[1]])
    for i,img in enumerate(gray_list):
        lapla_x = convolve2d(img, laplacian_x, mode='same', boundary='fill', fillvalue=np.average(img))
        lapla_y = convolve2d(img, laplacian_y, mode='same', boundary='fill', fillvalue=np.average(img))
        tmp = np.sqrt(lapla_x ** 2 + lapla_y ** 2)
        lapla_values[:,:,i] = gaussian_filter(tmp,50)#convolve2d(tmp, mv_kernel, mode='same', boundary='fill', fillvalue=np.average(tmp))

    #lapla_values = convolve2d(lapla_values, mv_kernel, mode='same', boundary='fill', fillvalue=np.average(lapla_values))
    img_map = np.argmax(lapla_values,axis=2)
    return img_map / np.max(img_map), img_map 
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
    images = {int(v.split('.')[0][5:]):v for i,v in enumerate(images)}
    images = dict(sorted(images.items())).values()
    rgb_list = []
    gray_list = []
    for img in images:
        img_path = os.path.join(focal_stack_dir, img)
        rgb_img = Image.open(img_path)
        gray_img = rgb_img.convert('L')
        rgb_list.append(np.array(rgb_img)/255)
        gray_list.append(np.array(gray_img)/255)
    return rgb_list, gray_list
    raise NotImplementedError



def onclick(event, rgb_list, depth_map, ax): 
    img_height, img_width, _ = rgb_list[0].shape
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        ax.imshow(rgb_list[depth_map[y,x]])
        plt.draw()
    else:
        sys.exit()

def refocusApp(rgb_list: List[np.ndarray], depth_map: np.ndarray) -> None:
    # Refocusing application
    # Input:
    #   rgb_list - List of RGB images for varying focal lengths
    #   depth_map - mxn index map
    #               depth_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)
    fig, ax = plt.subplots()
    ax.imshow(rgb_list[0])
    ax.set_title('Click on the image to get coordinates')
    ax.set_axis_off()
    fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, rgb_list, depth_map, ax))

    plt.show()

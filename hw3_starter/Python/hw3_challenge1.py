from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
def generateHoughAccumulator(edge_image: np.ndarray, theta_num_bins: int, rho_num_bins: int) -> np.ndarray:
    '''
    Generate the Hough accumulator array.
    Arguments:
        edge_image: the edge image.
        theta_num_bins: the number of bins in the theta dimension.
        rho_num_bins: the number of bins in the rho dimension.
    Returns:
        hough_accumulator: the Hough accumulator array.
    '''
    #print('here')
    #plt.figure()
    #plt.imshow(edge_image)
    #plt.show()
    theta = np.linspace(0, 180, theta_num_bins, endpoint=False)
    max_rho = np.sqrt(edge_image.shape[0]**2 + edge_image.shape[1]**2)
    #p = np.linspace(-max_rho, max_rho, rho_num_bins)

    accumulator = np.zeros((theta_num_bins, rho_num_bins))

    
    y_coords, x_coords = np.nonzero(edge_image)
    
    for i in range(len(y_coords)):
        y = y_coords[i]
        x = x_coords[i]
        
        for j in range(theta_num_bins):
            rho = x * np.cos(np.radians(theta[j])) + y * np.sin(np.radians(theta[j]))
            rho_index = int(np.round((rho + max_rho) / (2 * max_rho) * (rho_num_bins - 1)))
            accumulator[j][rho_index] += 1

    acc_min = np.min(accumulator)
    acc_max = np.max(accumulator)

    # Normalize to [0, 255]
    accumulator_normalized = 255 * (accumulator - acc_min) / (acc_max - acc_min)
    return accumulator_normalized


def lineFinder(orig_img: np.ndarray, hough_img: np.ndarray, hough_threshold: float):
    '''
    Find the lines in the image.
    Arguments:
        orig_img: the original image.
        hough_img: the Hough image.
        hough_threshold: the threshold for the Hough accumulator array.
    Returns: 
        line_img: PIL image with lines drawn.
    '''
    theta = np.linspace(0, 180, hough_img.shape[0])
    
    #p = np.linspace(-max_rho, max_rho, hough_img.shape[1])
    #print(theta)
    hough_peaks = np.argwhere(hough_img > hough_threshold)
    #plt.figure()
    #plt.imshow(hough_img)
    #plt.show()
    orig_height, orig_width = orig_img.shape
    rho_bins = hough_img.shape[1]

    max_rho = np.sqrt(orig_height**2 + orig_width**2)
    line_img = Image.fromarray(orig_img.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(line_img)
    #print(hough_peaks)
    #print(hough_peaks)
    for itheta, ip in hough_peaks:
        angle_rad = np.radians(theta[itheta])
        rho = (2 * ip - rho_bins)  * max_rho / rho_bins

        # Calculate coordinates of two points on the line
        if angle_rad != 0:
            x1, x2, x3 = 0, orig_width, -1 * orig_width
            y1 = (rho - x1 * np.cos(angle_rad)) / np.sin(angle_rad)
            y2 = (rho - x2 * np.cos(angle_rad)) / np.sin(angle_rad)
            y3 = (rho - x3 * np.cos(angle_rad)) / np.sin(angle_rad)
        else:
            x1, x2, x3 = rho, rho, rho
            y1, y2, y3 = orig_height, 0, -1 * orig_height
    
        draw.line((x1, y1, x2, y2, x3, y3), fill=(255,0,0), width=2)
    
    #line_img.show()
    return  line_img


    raise NotImplementedError

def lineSegmentFinder(orig_img: np.ndarray, edge_img: np.ndarray, hough_img: np.ndarray, hough_threshold: float):
    '''
    Find the line segments in the image.
    Arguments:
        orig_img: the original image.
        edge_img: the edge image.
        hough_img: the Hough image.
        hough_threshold: the threshold for the Hough accumulator array.
    Returns:
        line_segement_img: PIL image with line segments drawn.
    '''
    line_img = lineFinder(orig_img, hough_img, hough_threshold)
    line_img = np.array(line_img)
    width = line_img.shape[1]
    height = line_img.shape[0]


    structuring_element = np.ones((5, 5), dtype=bool)

    # Perform dilation
    edge_img = binary_dilation(edge_img, structure=structuring_element)
    for y in range(height-2):
        for x in range(width-2):
            if np.array_equal(line_img[y][x], [255,0,0]) and edge_img[y][x] == 0:
                line_img[y][x] = orig_img[y][x]
    return Image.fromarray(line_img)
    raise NotImplementedError


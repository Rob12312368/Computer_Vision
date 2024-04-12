import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
def onclick(event):
    # Check if the click happened within the axes of the image
    if event.inaxes:
        # Retrieve the coordinates (x, y) of the click
        x = event.xdata
        y = event.ydata
        print(f'Clicked at (x, y) = ({x:.2f}, {y:.2f})')
    else:
        sys.exit()


# Load and display an example image
img_path = 'data/stack/frame1.jpg'  # Specify the path to your image file
img = mpimg.imread(img_path)

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title('Click on the image to get coordinates')
ax.set_axis_off()

# Connect the onclick function to the mouse click event
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

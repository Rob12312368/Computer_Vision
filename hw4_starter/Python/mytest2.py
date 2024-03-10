from helpers import genSIFTMatches
from PIL import Image, ImageDraw

orig_img = Image.open('data/portrait.png').convert('RGB')
warped_img = Image.open('data/portrait_transformed.png').convert('RGB')
# x, y
start = [
        [162.0, 100.0],
        [644.0, 98.0],
        [640.0, 698.0],
        [162.0, 698.0]]

dest =   [
        [140.0, 144.0],
        [620.0, 28.0],
        [664.0, 770.0],
        [116.0, 592.0]]
draw = ImageDraw.Draw(orig_img)
drawtwo = ImageDraw.Draw(warped_img)

radius = 5  # Adjust this value to your desired size
x = 0
y = 0
# Draw the point with the specified radius
for i in range(len(start)):
    draw.ellipse((start[i][0] - radius, start[i][1] - radius, start[i][0] + radius, start[i][1] + radius), fill=(255,0,0))
    drawtwo.ellipse((dest[i][0] - radius, dest[i][1] - radius, dest[i][0] + radius, dest[i][1] + radius), fill=(255,0,0))
orig_img.show()
warped_img.show()


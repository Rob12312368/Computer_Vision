1b Voting Scheme: I used 2100 bins for p, and 240 bins for theta. I chose those two values based on experiments. I started with theat=180 and p = sqrt(height^2+width^2).
1c Peaks: I used thresholds to find peaks. I try to see the bigger values in the hough transform matrix, and chose a reasonable threshold.
1d Algorithm: I go through each pixel of the lines and see if its edge pixel is white. If it is white, then it should be the segment. Also, I used dilation to expand the line a bit.

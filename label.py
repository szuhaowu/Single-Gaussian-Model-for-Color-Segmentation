
import imageio
import numpy as np
from matplotlib import pyplot as plt

from roipoly import RoiPoly

# Create image
for i in range(1,2):
    x = 'C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/trainset/{}.png'.format(i)
    img = imageio.imread(x)

    # Show the image
    fig = plt.figure()
    plt.imshow(img, interpolation='nearest', cmap="Greys")
    plt.colorbar()
    plt.title("left click: line segment         right click: close region")
    plt.show(block=False)

    # Let user draw first ROI
    roi1 = RoiPoly(color='r', fig=fig)

    # Show the image with the first ROI
    fig = plt.figure()
    plt.imshow(img, interpolation='nearest', cmap="Greys")
    plt.colorbar()
    roi1.display_roi()
    plt.title('draw second ROI')
    plt.show(block=False)
    
    # Let user draw second ROI
    roi2 = RoiPoly(color='b', fig=fig)
    img= img[:,:,1]
    # Show the image with both ROIs and their mean values
    #plt.imshow(img, interpolation='nearest', cmap="Greys")
    #plt.colorbar()
    #[x.display_roi() for x in [roi1, roi2]]
    #[x.display_mean(img) for x in [roi1, roi2]]
    #plt.title('The two ROIs')
    #plt.show()

# Show ROI masks
    plt.imshow(roi1.get_mask(img) + roi2.get_mask(img),
           interpolation='nearest', cmap="Greys")
    plt.title('ROI masks of the two ROIs')
    plt.show()
    np.save('otherblue{}.npy'.format(i), roi1.get_mask(img) + roi2.get_mask(img))
    
    
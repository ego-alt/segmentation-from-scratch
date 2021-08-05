from labelling import ImageProcessor
from skimage.measure import label


root = ' '
save = ' '

process = ImageProcessor(root, save)
names, layered_images, num = process.handle()

for i, img in enumerate(layered_images):
    output = label(img)
    process.save_im(names[i], output, num[i])
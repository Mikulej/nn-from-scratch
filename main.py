from datasets import load_dataset
from datasets import load_dataset_builder

from Layer import *
import numpy as np
def convertToPixels(image) -> []:
    width, height = image.size
    rgb_im = image.convert('RGB')
    pixels = []
    for x in range(width):
        for y in range(height):
            r, g, b = rgb_im.getpixel((x, y))
            pixels.append([r,g,b])
    return pixels

def convertToPixelsMono(image) -> []:
    width, height = image.size
    rgb_im = image.convert('RGB')
    pixels = []
    for x in range(width):
        for y in range(height):
            r, g, b = rgb_im.getpixel((x, y))
            pixels.append(r)
    return pixels

def L_i_vectorized(x,y,W):
    scores = np.dot(W,x)
    margins = np.maximum(0,scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i

#Initialize weights
weights = np.ndarray(784*10)
for i in weights:
    i = random.uniform(-1.0,1.0) 
#layer = Layer(784)

#get info about dataset
# ds = load_dataset_builder("ylecun/mnist")
# print(ds.info.description)
# print(ds.info.features)

# ds = load_dataset("ylecun/mnist", split="train")
# #ds = load_dataset("ylecun/mnist", split="test")
# print(ds)
# #get label
# print(ds[0]["label"])
# #get rgb_array/monochrome_array
# image = ds[1]["image"]
# print("RGB:")
# pixels = convertToPixels(image)
# print(pixels)
# print("Mono:")
# pixelsMono = convertToPixelsMono(image)
# print(pixelsMono)
# #get row
# print(ds[2])




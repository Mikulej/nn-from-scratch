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

def sigmoid(x):
  if x > 0:   
    z = np.exp(-x)
    return 1/(1+z)
  else:
    z = np.exp(x)
    return z/(1+z)
  
#CS231n example 2 layer network
# X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# y = np.array([[0,1,1,0]]).T
# syn0 =  2*np.random.random((3,4)) - 1
# syn1 =  2*np.random.random((4,1)) - 1
# for j in range(6000):
#     #print(j)
#     l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
#     l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
#     l2_delta = (y- l2) *(l2*(1-l2))
#     l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
#     syn1 += l1.T.dot(l2_delta)
#     syn0 += X.T.dot(l1_delta)


#layer = Layer(784)

#get info about dataset
# ds = load_dataset_builder("ylecun/mnist")
# print(ds.info.description)
# print(ds.info.features)

ds = load_dataset("ylecun/mnist", split="train")
#ds = load_dataset("ylecun/mnist", split="test")
print(ds)
#get label
# print("Label of image nr0 is: ", ds["label"][0])
# #get rgb_array/monochrome_array
# image = ds["image"][0]
# print("RGB:")
# pixels = convertToPixels(image)
# print(pixels)
# print("Mono:")
# pixelsMono = convertToPixelsMono(image)
# print(pixelsMono)
# #get row
# print(ds[2])


IMAGE_AMOUNT = 10
#Initialize weights
weights = np.ndarray((784,IMAGE_AMOUNT),dtype=np.longdouble)#array 78400x1 (1-dimensional data is Monochrome) (3-diemnsional data is RGB)

for i in weights:
    i = random.uniform(1.0,5.0) 

#weights = 2 * np.random.random((784,IMAGE_AMOUNT)) - 1

#Store images in X (inputs)
#Store labels of images in y (labels/expected output of classifier)
X = []
y = []
iter = ds.iter(batch_size=1)
j = 0
for i in iter:
    #print(i["label"][0])
    print(j)
    X.append(convertToPixelsMono(i["image"][0]))
    y.append(i["label"][0])
    j += 1
    if j == IMAGE_AMOUNT:
        break
X = np.array(X,dtype=np.longdouble)
y = np.array(y)

print(X.shape)
print(weights.shape)
#Train
for j in range(10):
    #X = convertToPixelsMono(ds[1]["image"])
    #forward, compute first layer activations (sigmoid)
    #sigmoided = np.ndarray((784,784))
    dot_product = np.dot(X,weights)
    print(dot_product)
    print(dot_product.shape)
    for k in dot_product:
        k = sigmoid(k)
    l1 = dot_product
    #l1 = 1/(1+np.exp(-(np.dot(X,weights))))
    #backward
    l1_delta = (y - l1) * (l1 * (1-l1))
    #update
    weights += X.T.dot(l1_delta)


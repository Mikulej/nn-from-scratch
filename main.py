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
  
def svm_loss(label_index: int,output) -> int:
    loss_vector: list = []
    i: int = 0
    for row in output:   
        loss: np.float64 = 0
        j: int = 0
        for c in row:
            if label_index == j:
                j += 1
                continue
            value: np.float64 = np.maximum(0.0,c-row[label_index]+1.0)
            loss += value
            j += 1
        loss_vector.append(value)
        i += 1
    return np.array(loss_vector)
  

#get info about dataset
# ds = load_dataset_builder("ylecun/mnist")
# print(ds.info.description)
# print(ds.info.features)

ds = load_dataset("ylecun/mnist", split="train")
ds_test = load_dataset("ylecun/mnist", split="test")
print(ds)
#get label
print("Label of image nr0 is: ", ds["label"][0])
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


IMAGE_AMOUNT = 100
#Initialize weights
weights = 2 * np.random.random((784,10)) - 1 # get number from [-1;1]

#Store images in X (inputs)
#Store labels of images in y (labels/expected output of classifier)
X = []
y = []
iter = ds.iter(batch_size=1)
j = 0
for i in iter:
    #print(j)
    X.append(convertToPixelsMono(i["image"][0]))
    y.append(i["label"][0])
    j += 1
    if j == IMAGE_AMOUNT:
        break
X = np.array(X,dtype=np.float64)
y = np.array(y,dtype=np.float64)

print("X,shape=",X.shape)
print("weights.shape=",weights.shape)
print("y.shape=",y.shape)
# print("X=",X)
# print("weights=",weights)

#Train
for iterations in range(1):
    #forward, compute first layer activations (sigmoid)
    # dot_product = np.dot(X,weights)
    # for row in dot_product:
    #     for k in row:
    #         k = sigmoid(k)
    # l1 = dot_product
    # l1 = 1/(1+np.exp(-(np.dot(X,weights))))
    # # backward (backpropagation) (sigmoid)
    # l1_delta = (y - l1) * (l1 * (1-l1))
    #forward, compute first layer activations (ReLU)
    dot_product = np.dot(X,weights)
    #print("dot_product=",dot_product)
    i: int = 0
    l1_delta = np.zeros_like(dot_product)#for backward (backpropagation) (ReLU)
    print(l1_delta)
    for row in dot_product:
        j: int = 0
        for k in row:
            k = np.maximum(0,k)
            if k > 0:
                l1_delta[i][j] = 1
            j+=1
        i+=1
    l1 = dot_product

    #update (weights)
    weights += X.T.dot(l1_delta)
    #calculate loss
    #print("Loss for label 5 =",svm_loss(5,l1))
    loss = svm_loss(5,l1)
    print("loss=", loss)
    print("loss.shape=", loss.shape)
    #loss = #L_i_vectorized(X.T,5,weights.T)
    #print("loss=", loss)
    # print("l1=",l1)
    # print("l1.shape=",l1.shape)


#Test


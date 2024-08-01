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
    loss: np.float64 = 0 # |  ||
                         # || |_
    i: int = 0
    output = output.T
    for row in output:
        for c in row:
            print(c)
        #print(row)
        if label_index == i:
            i += 1
            continue
        value: np.float64 = max(0,row[i]-row[label_index]+1.0)
        loss += value
        print("Loss =",loss," value =",value)
        i += 1
    return loss
  
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
ds_test = load_dataset("ylecun/mnist", split="test")
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


IMAGE_AMOUNT = 15
#Initialize weights
weights = np.ndarray((784,10),dtype=np.float64)#array 784(pixels aka input)x10(10 labels 0..9) (1-dimensional data is Monochrome) (3-diemnsional data is RGB)

for i in weights:
    i = random.uniform(-10.0,10.0) 

#weights = 2 * np.random.random((784,IMAGE_AMOUNT)) - 1

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
# for i in X:
#     print(i)
print("weights.shape=",weights.shape)
#print(weights)
print("y.shape=",y.shape)
#Train
for iterations in range(10):
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
    i: int = 0
    l1_delta = np.zeros_like(dot_product)#for backward (backpropagation) (ReLU)
    print(l1_delta)
    for row in dot_product:
        j: int = 0
        for k in row:
            k = max(0,k)
            if k > 0:
                l1_delta[i][j] = 1
            j+=1
        i+=1
    l1 = dot_product
    
    
    
    
    #update (weights)
    weights += X.T.dot(l1_delta)
    #calculate loss
    print("l1=",l1)
    print("l1.shape=",l1.shape)
    print("Loss for label 0 =",svm_loss(5,l1))
    
#so dot(X,weights) is labels 
#   0 1 2 3 4 5 6 7 8 9 (labels)
# 0
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9 (scores)
#Its possible to calculate SVM_loss using this ^

# for row in weights:
#     for i in row:
#         print(i)

#Test


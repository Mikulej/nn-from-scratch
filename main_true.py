from datasets import load_dataset
import numpy as np

def convertToPixelsMono(image) -> []:
    width, height = image.size
    rgb_im = image.convert('RGB')
    pixels = []
    for x in range(width):
        for y in range(height):
            r, g, b = rgb_im.getpixel((x, y))
            r /= 255
            pixels.append(r)
    return pixels

ds_train = load_dataset("ylecun/mnist", split="train")

#Get Data
BATCH_SIZE = 32 #aka Image amount
train_X = []
train_y = []
iter = ds_train.iter(batch_size=1)
j = 0
for i in iter:
    #print(j)
    train_X.append(convertToPixelsMono(i["image"][0]))
    train_y.append(i["label"][0])
    j += 1
    if j == BATCH_SIZE:
        break
train_X = np.array(train_X,dtype=np.longdouble)
train_y = np.array(train_y,dtype=np.longdouble)

def init_weights():
    W1 = 2 * np.random.random((784,10)) - 1
    W2 = 2 * np.random.random((10,10)) - 1
    return W1, W2

def ReLU(Z):
    return np.maximum(0,Z)

# def softmax(Z):
#     return np.exp(Z / np.sum(np.exp(Z)))

def softmax(Z):
    #get matrix, return matrix of probabilites
    S = np.zeros_like(Z)
    # print("Z=",Z)
    # print("Z.shape=",Z.shape)
    i: int = 0
    for row in Z: #for each image
        j: int = 0
        for c in row: #for each score
            S[i][j] = np.exp(c) / np.sum(np.exp(row))
            j += 1
        i += 1
    # print("S=",S)
    # print("S.shape=",S.shape)

    return S

def forward(X,W1,W2):
    Z1 = np.dot(X,W1) #(32, 784)x(784, 10)=(32, 10)
    # print("X.shape=",X.shape)
    # print("W1.shape=",W1.shape)
    # print("Z1.shape=",Z1.shape)
    A1 = ReLU(Z1) # (32, 10)
    #print("A1.shape=",A1.shape)
    Z2 = np.dot(A1,W2) #(32, 10)x(10, 10)=(32, 10)
    #print("Z2.shape=",Z2.shape)
    A2 = softmax(Z2)
    # for row in A2:
    #     print("np.sum(row)=",np.sum(row))
    #     for c in row:
    #         print(c)
    #print("A2.shape=",A2.shape)
    return Z1,A1,Z2,A2
    
    

def gradient_descent(X,Y,iterations,alpha):
    W1, W2 = init_weights()
    for i in range(iterations):
        Z1,A1,Z2,A2 = forward(X,W1,W2)
        # backward()
        # update(W1)

gradient_descent(train_X,train_y,5,0.1)
print("End")
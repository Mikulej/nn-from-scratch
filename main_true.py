from datasets import load_dataset
import numpy as np

def convertToPixelsMono(image) -> []:
    width, height = image.size
    rgb_im = image.convert('RGB')
    pixels = []
    for x in range(width):
        for y in range(height):
            r, g, b = rgb_im.getpixel((x, y))
            r /= 255.0
            pixels.append(r)
    return pixels

BATCH_SIZE = 32 #aka Image amount
#ds_train = load_dataset("ylecun/mnist", split="train",cache_dir="cache")
ds_train = load_dataset("ylecun/mnist", split="train")

def get_X_from_batch(batch):
    X = []
    print("batch=",batch)
    for image in batch:
        X.append(convertToPixelsMono(image))
    return np.array(X)

def get_Y_from_batch(batch):
    return np.array(batch)

def init_weights():
    W1 = 2 * np.random.random((784,10)) - 1
    W2 = 2 * np.random.random((10,10)) - 1
    return W1, W2

def ReLU(Z):
    return np.maximum(0,Z)

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

def one_hot(Y):
    one_hot_Y = np.zeros((BATCH_SIZE,10))
    i: int = 0
    # print("Y=",Y)
    for row in one_hot_Y:
        row[Y[i]] = 1
        i += 1
    # print("one_hot_Y=",one_hot_Y)
    # print("one_hot_Y.shape=",one_hot_Y.shape)
    return one_hot_Y

    
    
def backward(Z1,A1,Z2,A2,W1,W2,X,Y):
    one_hot_Y = one_hot(Y)
    dZ1 = A2 - one_hot_Y
    #print(Z2)


def gradient_descent(epochs,alpha):
    W1, W2 = init_weights()
    for epoch in range(epochs):
        batch = ds_train.shuffle()
        batch = batch.flatten_indices()
        #print("range(len(batch)/BATCH_SIZE)=",len(batch)/BATCH_SIZE)
        for i in range(int(len(batch)/BATCH_SIZE)):
            X = get_X_from_batch(batch["image"][i:i+BATCH_SIZE])
            Y = get_Y_from_batch(batch["label"][i:i+BATCH_SIZE])
            print("Y=",Y)
            print("Y.shape=",Y.shape)
            print("X=",X)
            print("X.shape=",X.shape)
            Z1,A1,Z2,A2 = forward(X,W1,W2)
            backward(Z1,A1,Z2,A2,W1,W2,X,Y)
            # update(W1)
            print("Next Batch...")

gradient_descent(1,0.1)
print("End")
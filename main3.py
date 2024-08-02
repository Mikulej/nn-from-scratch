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
IMAGE_AMOUNT = 10
X = []
y = []
iter = ds_train.iter(batch_size=1)
j = 0
for i in iter:
    #print(j)
    X.append(convertToPixelsMono(i["image"][0]))
    y.append(i["label"][0])
    j += 1
    if j == IMAGE_AMOUNT:
        break
X = np.array(X,dtype=np.longdouble)
y = np.array(y,dtype=np.longdouble)

#Initialize Params weights and biases
def init_params():
    W1 = 2*np.random.random((784,10))-1
    W2 = 2*np.random.random((10,10))-1
    return W1,W2

def ReLU(Z):
    return np.maximum(0,Z)

def softmax(Z):
    # for i in Z:
    #     for j in i:
    #         print(j)
    A = np.exp(Z / np.sum(np.exp(Z)))
    # for i in A:
    #     for j in i:
    #         print(j)
    return A

def forward(X,W1,W2):
    Z1 = np.dot(X,W1)
    #print("Z1=",Z1)
    A1 = ReLU(Z1)
    #print("A1=",A1)
    Z2 = np.dot(W2,A1)
    #print("Z2=",Z2)
    A2 = softmax(Z2)
    #print("A2=",A2)
    # print("X.shape=",X.shape)
    # print("W1.shape=",W1.shape)
    # print("Z1.shape=",Z1.shape)
    # print("A1.shape=",A1.shape)
    # print("Z2.shape=",Z2.shape)
    # print("A2.shape=",A2.shape)
    return Z1,A1,Z2,A2

def one_hot(Y):
    #print(Y)
    one_hot_Y = np.zeros((Y.size,int(Y.max())+1))
    one_hot_Y[np.arange(Y.size), Y.astype(int)] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def dReLU(Z):
    return Z > 0

def backward(Z1,A1,Z2,A2,W2,X,Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m* dZ2.dot(A1.T)
    dZ1 = W2.T.dot(dZ2) * dReLU(Z1)
    dW1 = 1/m* np.dot(X.T,dZ1)
    # print("START")
    # print("dZ2.shape=",dZ2.shape)
    # print("A2.shape=",A2.shape)
    # print("one_hot_Y.shape=",one_hot_Y.shape)
    
    # print("dW2.shape=",dW2.shape)
    # print("dZ2.shape=",dZ2.shape)
    # print("A1.T.shape=",A1.T.shape)

    # print("dZ1.shape=",dZ1.shape)
    # print("dW2.T.shape=",dW2.T.shape)

    # print("dW1.shape=",dW1.shape)
    # print("X.shape=",X.shape)
    # print("END")
    
    
    return dW1,dW2

def update_params(W1,W2,dW1,dW2,alpha):
    W1 = W1 - alpha * dW1
    W2 = W2 - alpha * dW2
    return W1,W2

def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions == Y)/ Y.size

def gradient_descent(X,Y,iterations,alpha):
    W1,W2 = init_params()
    # print("W1=",W1)
    # print("W2=",W2)
    for i in range(iterations):
        Z1, A1 ,Z2 ,A2 = forward(X,W1,W2)
        dW1, dW2 = backward(Z1,A1,Z2,A2,W2,X,Y)
        W1,W2=update_params(W1,W2,dW1,dW2,alpha)
        if i % 1 == 0:
            print("Iteration", i)
            acc = get_accuracy(get_predictions(A2), Y)
            print("Accuracy:",acc)
            
    return W1,W2


W1, W2 = gradient_descent(X,y,300,0.1)

from datasets import load_dataset
import numpy as np

def convertToPixelsMono(image) -> []:
    width, height = image.size
    rgb_im = image.convert('RGB')
    pixels = []
    for x in range(width):
        for y in range(height):
            r, g, b = rgb_im.getpixel((x, y))
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
X = np.array(X,dtype=np.float64)
y = np.array(y,dtype=np.float64)

#Initialize Params weights and biases
def init_params():
    W1 = np.random.random((784,10))
    b1 = np.random.random((10,1))
    W2 = np.random.random((10,10))
    b2 = np.random.random((10,1))
    return W1,b1,W2,b2

def ReLU(Z):
    return np.maximum(0,Z)

def softmax(Z):
    np.exp(Z / np.sum(np.exp(Z)))

def forward(X,W1,b1,W2,b2):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1,A1,Z2,A2

def one_hot(Y: np.array):
    one_hot_Y = np.zeros((Y.size,Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def dReLU(Z):
    return Z > 0

def backward(Z1,A1,Z2,A2,X,Y):
    m = y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m* dZ2.dot(A1.T)
    db2 = 1 /m* np.sum(dZ2,2)
    dZ1 = W2.T.dot(dZ2) * dReLU(Z1)

    dW1 = 1/m* dZ1.dot(X.T)
    db1 = 1 /m* np.sum(dZ1,2)
    return dW1,db1,dW2,db2

def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1,b1,W2,b2

def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions == Y)/ Y.size

def gradient_descent(X,Y,iterations,alpha):
    W1,b1,W2,b2 = init_params()
    for i in range(iterations):
        Z1, A2 ,Z2 ,A2 = forward(X,W1,b1,W2,b2)
        dW1, db1, dW2, db2 = backward(Z1,A2,Z2,A2,W2,X,Y)
        W1,b1,W2,b2=update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha)
        if i % 50 == 0:
            print("Iteration", i)
            print("Accuracy ", get_accuracy(get_predictions(A2), Y))
    return W1,b1,W2,b2


W1, b1, W2, b2 = gradient_descent(X,y,500,0.1)

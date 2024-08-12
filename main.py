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
    #print("batch=",batch)
    for image in batch:
        X.append(convertToPixelsMono(image))
    return np.array(X)

def get_Y_from_batch(batch):
    return np.array(batch)

def init_weights():
    W1 = 2 * np.random.random((784,10)) - 1
    b1 = 2 * np.random.random((1,10)) - 1
    W2 = 2 * np.random.random((10,10)) - 1
    b2 = 2 * np.random.random((1,10)) - 1
    return W1, b1, W2, b2

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

def cross_entropy_loss(predictions,labels):
    loss = []
    for prow ,lrow in zip(predictions, labels):
        loss.append(-np.sum(lrow * np.log(prow)))
    loss = np.array(loss)
    # print("loss=", loss)
    # print("loss.shape=", loss.shape)
    return loss
            
            




def forward(X,W1, b1, W2, b2,Y):
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
    # print("A2.shape=",A2.shape)
    loss = cross_entropy_loss(A2,one_hot(Y))
    return Z1,A1,Z2,A2,loss

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
    # print("A2=",A2)
    # print("A2.shape=",A2.shape)
    #dZ2 = cross_entropy_loss(A2,one_hot_Y)
    dZ2 = A2 - one_hot_Y
    
    #m = Y.shape[0]
    dW2 = dZ2.T.dot(A1).T # (10, 10) = (10, 32) x (32, 10)
    # print("dZ2.shape=",dZ2.shape)
    # print("A1.shape=",A1.shape)
    # print("dW2.shape=",dW2.shape)
    dA1 = dZ2.dot(W2.T) # (32, 10) = (32, 10) x (10, 10)
    #print("dA1.shape=",dA1.shape)

    dZ1 = dA1 * (Z1 > 0) # (32, 10) = (32, 10) x (32, 10)
    # print("dZ1.shape=",dZ1.shape) # (32, 10)
    # print("dA1.shape=",dA1.shape) # (32, 10)
    # print("Z1.shape=",Z1.shape) # (32, 10)

    dW1 = dZ1.T.dot(X).T
    # print("dW1.shape=",dW1.shape) # (784, 10) 
    # print("X.shape=",X.shape) # (32, 784)
    #gpt start
    # m = Y.shape[0]
    # dW2 = np.dot(A1.T, dZ2) / m 
    # dA1 = np.dot(dZ2, W2.T) 

    # dZ1 = dA1 * (Z1 > 0)
    # dW1 = np.dot(X.T, dZ1) / m 
    #gpt end
    return dW1,dW2

def update(W1,W2,dW1,dW2,alpha):
    W1 += -dW1 * alpha
    W2 +=  -dW2 * alpha
    return W1, W2

def predict(Z,Y):
    correct: int = 0
    i: int = 0
    for row in Z:
        print("Label=",Y[i]," predicted=",np.argmax(row)," acc=",np.max(row))
        if Y[i] == np.argmax(row):
            correct += 1
        #predictions.append(np.maximum(row),Y[i])
        i += 1
    print("Correct guessed=",correct,"/",BATCH_SIZE)

def gradient_descent(epochs,alpha):
    W1,b1, W2, b2 = init_weights()
    for epoch in range(epochs):
        batch = ds_train.shuffle()
        batch = batch.flatten_indices()
        #print("range(len(batch)/BATCH_SIZE)=",len(batch)/BATCH_SIZE)
        for i in range(int(len(batch)/BATCH_SIZE)):
            X = get_X_from_batch(batch["image"][i:i+BATCH_SIZE])
            Y = get_Y_from_batch(batch["label"][i:i+BATCH_SIZE])
            # print("Y=",Y)
            # print("Y.shape=",Y.shape)
            # print("X=",X)
            # print("X.shape=",X.shape)
            Z1,A1,Z2,A2,loss = forward(X,W1,b1,W2,b2,Y)
            dW1, dW2 = backward(Z1,A1,Z2,A2,W1,W2,X,Y)
            W1, W2 = update(W1,W2,dW1,dW2,alpha)
            print("Losses=",loss)
            predict(A2,Y)

gradient_descent(1,0.03)
print("End")
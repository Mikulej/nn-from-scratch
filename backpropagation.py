from micrograd.engine import Value
import random
#init
learning_rate = 0.1
x = random.random()
w = Value(2.0)
b = Value(3.0)
y = 10.0
for i in range(100):
    
    #forward
    z1 = x * w
    z2 = z1 + b
    #calculate loss
    z3 = y-z2
    loss = z3**2
    #print(i,loss.data)
    
    #backward
    w.grad = 0.0
    b.grad = 0.0
    loss.backward()

    #update
    w.data += -w.grad * learning_rate
    b.data += -b.grad * learning_rate

    #teach ml that z=10
    print(z2)
    #show gradients
    print("loss",loss.grad)
    print("z3",z3.grad, z3.data)
    print("z2",z2.grad)
    print("z1",z1.grad)
    print("w",w.grad)
    



    

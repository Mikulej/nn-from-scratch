# Neural Network From Scratch

Personal project with simple objective:   
Learn deep learning by creating neural network capable of image detection.  
Without using libraries like PyTorch or TensorFlow!  
Dataset https://huggingface.co/datasets/ylecun/mnist

Steps:
- Initialize weights DONE
- Do forward pass DONE
- Do backpropagation DONE
- Update weigths DONE
- Loop this by giving another batch from X (total training data) DONE

Thoughts:
Linear algebra is **R**eally **C**ool!
Requirement for dot product of `m1 x m2` is `m1.numberOfcolumns==m2.numberOfrows`
```py
m1 = np.ndarray((10,784))
m2 = np.ndarray((784,1))
print(np.dot(m1,m2).shape)

>>> (10,10)
```

"A.shape will return a tuple (m, n), where m is the number of rows, and n is the number of columns."

When your neural network ends with ...->normalization->loss function
Calculating gradient starts from loss function **NOT** normalization

Backpropagation might be hard to understand but when you do understand it, it becomes quite easy as a concept.

Softmax + Cross Entropy Loss = Good (Easy deriative calculation)

Sometimes matrix (N, N) might be a problem, try to transpose it!  

Useful sources:  
https://www.parasdahal.com/softmax-crossentropy  
https://youtu.be/VMj-3S1tku0?si=WmW3sketuuTzM0qv  
https://github.com/karpathy/micrograd
# Requirements
Python 3.11
[Miniconda](https://docs.anaconda.com/free/miniconda/index.html)
`pip install -r requirements.txt`

# About Backpropagation
We represent each weight as a tuple/class of data/value it holds and "global" gradient.
"Global" gradient tell us in which direction shifting data by small amount will **increase** the **final** output.

In `backpropagation.py` example has:
- `x` - input
- `w` - weight
- `b` - bias
- `y` - label (we want to teach network that f(x)=y)


Notation dx/dy means deriative of x with respect to y:
e.g. `x = 2y^3 + z, dx/dx = ? dx/dy = ?, dx/dz = ?`  
`dx/dy = 2*3y^2 + 0`
`dx/dy = 6y^2`

`dx/dz = 0 + 1`
`dx/dz = 1`

`dx/dx = 1 always`

Backpropagation starts from the end of the network:
1. Squared loss, gradient = *dL/dL* = 1 (last layer always has gradient = 1)
2. `L=z3**2` gradient = *dL/dz3* = `2*z3` (z3 represents data/value that z3 holds)
Let's say `z3.data=1.27` then `z3.grad=2*z3.data` = `z3.grad=2*1.27` = `z3.grad=2.54`
so if we increase z3 it will increase output, since the last layer is loss we want to
minimalize it, that's why in update we do negate the gradient e.g `w.data += -w.grad * learning_rate`
3. `z3 = y-z2` gradient = *dL/dy* = `dL/z3*z3/y` (we apply chain rule here)
To get global gradient for y we need to multiply global gradient from output with local gradient  
We take global gradient from 1 step above like here from step 2. *dL/dz3* = `z3.grad=2.54`  
Local gradient is *dz3/dy* = `1+0` = `1` and *dz3/dz2* = `0-1` = `-1`  
`global_gradient` = `global_gradient_from_parent_operation*local_gradient`  
*dL/dy* = `z3.grad=2.54` * `dz3/dy=1` = `y.grad=2.54`
*dL/dz2* = `z3.grad=2.54` * `dz3/dz2=-1` = `z2.grad=-2.54`
y doesn't need gradient anyway since it's a label, we can't control it
4. `z2 = z1 + b`  
*dL/dz1* = `z2.grad=-2.54` * `dz2/dz1=1` = `z1.grad=-2.54`  
*dL/db* = `z2.grad=-2.54` * `dz2/db=1` = `b.grad=-2.54`  
5. `z1 = x * w` (lets assume x = 2)  
*dL/dw* = `z1.grad=-2.54` * `dz1/dw=x=2` = `w.grad=-5.08`

Now we know in which direction we should shift weight and bias to minimalize the loss
Remeber, gradient shows in which direction final output (aka Loss) **increases**.
We negate gradient and shift weights accordingly, `learning_rate` is a hyperparameter
```
    w.data += -w.grad * learning_rate
    b.data += -b.grad * learning_rate
```

Neural networks are build in a way that weights are represented by neurons that are stored in layers. Few layers can connect with each other creating full neural network.

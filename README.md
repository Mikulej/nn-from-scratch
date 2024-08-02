# Neural Network From Scratch

Personal project with simple objective:   
Learn deep learning by creating neural network capable of image detection.  
Without using libraries like PyTorch or TensorFlow!  
Dataset https://huggingface.co/datasets/ylecun/mnist

Thoughts:
Linear algebra is **R**eally **C**ool!
But this is not...
```py
m1 = np.ndarray((10,784))
m2 = np.ndarray((784,1))
print(np.dot(m1,m2).shape)

>>> (10,10)
```
For some reason it's **C**ool **R**eally


# Requirements
Python 3.11
[Miniconda](https://docs.anaconda.com/free/miniconda/index.html)
`pip install -r requirements.txt`
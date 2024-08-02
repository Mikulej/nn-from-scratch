# Neural Network From Scratch

Personal project with simple objective:   
Learn deep learning by creating neural network capable of image detection.  
Without using libraries like PyTorch or TensorFlow!  
Dataset https://huggingface.co/datasets/ylecun/mnist

Steps:
- Initialize weights DONE
- Do forward pass DONE
- Do backpropagation
- Update weigths
- Loop this by giving another batch from X (total training data), (should I change batch of training data each time I do one step? (forward+backward+update) or should I keep passing the same batch a few times)

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

"A.shape will return a tuple (m, n), where m is the number of rows, and n is the number of columns."

# Requirements
Python 3.11
[Miniconda](https://docs.anaconda.com/free/miniconda/index.html)
`pip install -r requirements.txt`
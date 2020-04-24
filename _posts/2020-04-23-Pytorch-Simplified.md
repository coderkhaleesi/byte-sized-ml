---
layout: post
title: "Simplifying Pytorch"
subtitle: "A Pytorch Notebook in Progess for Deep Learning"
date: 2020-04-23 12:34:07 -0400
background: '/img/posts/07.jpg'
---

# A notebook to learn all the basic pytorch that's needed

Jupyter Lab Shortcuts

Shift + Tab (Windows) to see a function params and definition

References

https://github.com/ritchieng/the-incredible-pytorch

https://github.com/yunjey/pytorch-tutorial

https://github.com/vahidk/EffectivePyTorch

https://github.com/dsgiitr/d2l-pytorch

https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#module-numpy.doc.broadcasting


```python
import torch
import numpy as np
```

## Basics


```python
#Conversion from numpy array to torch tensor and vice versa
tensor_0 = torch.Tensor([1,2])
arr = np.random.rand(2,2)
tensor_1 = torch.from_numpy(arr)

print(type(arr))
print(type(tensor_1))
```

    <class 'numpy.ndarray'>
    <class 'torch.Tensor'>



```python
torch_tensor = torch.ones(2,2)
type(torch_tensor)
```




    torch.Tensor




```python
numpy_arr = torch_tensor.numpy()
print(type(numpy_arr))
```

    <class 'numpy.ndarray'>



```python
#CPU vs GPU
tensor_init = torch.rand(2,2) #this is on CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tensor_init.to(device) #to GPU
tensor_init.cpu() #to back to CPU
```




    tensor([[0.6775, 0.7982],
            [0.4669, 0.1016]])




```python
c = torch.add(tensor_init, torch_tensor)
print(c)
c.add_(c)
```

    tensor([[1.6775, 1.7982],
            [1.4669, 1.1016]])





    tensor([[3.3550, 3.5964],
            [2.9338, 2.2032]])




```python
c=torch.sub(tensor_init,c)
print(c)
c.sub_(c) #only this is in place (_)
print(c.sub(torch.ones(2,2))) #not in place
print(c)
```

    tensor([[0.6775, 0.7982],
            [0.4669, 0.1016]])
    tensor([[-1., -1.],
            [-1., -1.]])
    tensor([[0., 0.],
            [0., 0.]])



```python
#element wise mul
c=torch.rand(2,1).mul(torch.rand(2,1))
print(c) #similarly div for division, also mul_ and div_
```

    tensor([[0.0195],
            [0.0051]])



```python
c.size()
print(c)
print(c.long())
print(c)
```

    tensor([[0.0195],
            [0.0051]])
    tensor([[0],
            [0]])
    tensor([[0.0195],
            [0.0051]])



```python
a = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
a.size()
```




    torch.Size([2, 10])




```python
a.mean(dim=1) #dim in pytorch is same as axis in numpy
```




    tensor([5.5000, 5.5000])




```python
a.std() #np.cov can be used if you want to calculate covariace matrix
```




    tensor(2.9469)




```python
#view
```

## Autograd

A torch.Tensor if has .requires_grad set as True, all the operations on that tensor will be tracked.

Call .backward() after finishing all operations to compute the gradients and .grad attribute saves all the gradients with respect to other vars.

with torch.no_grad():
for when we don't want to track

Each torch.Tensor has a Function (.grad_fn) which has created the Tensor as a result of operations defined in the computational graph.

Note: if your Tensor is multi-dimensional, need to specify a gradient argument that is a tensor of matching shape


```python
from torchviz import make_dot #open terminal and install if using jupyter lab, much easier than notebook
```


```python
x = torch.tensor([[1.,2.], [3.,4.]], requires_grad=True)
print(x)

```

    tensor([[1., 2.],
            [3., 4.]], requires_grad=True)


Note: See in the next cell the error when we try to compute using numpy

This is expected behavior because moving to numpy will break the graph and so no gradient will be computed. If you don’t actually need gradients, then you can explicitly .detach() the Tensor that requires grad to get a tensor with the same content that does not require grad. This other Tensor can then be converted to a numpy array.


```python
y = np.power(np.linalg.norm(x), 2) + 4
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-40-eea598aa356e> in <module>
    ----> 1 y = np.power(np.linalg.norm(x), 2) + 4


    <__array_function__ internals> in norm(*args, **kwargs)


    ~\Anaconda3\lib\site-packages\numpy\linalg\linalg.py in norm(x, ord, axis, keepdims)
       2458
       2459     """
    -> 2460     x = asarray(x)
       2461
       2462     if not issubclass(x.dtype.type, (inexact, object_)):


    ~\Anaconda3\lib\site-packages\numpy\core\_asarray.py in asarray(a, dtype, order)
         83
         84     """
    ---> 85     return array(a, dtype, copy=False, order=order)
         86
         87


    ~\Anaconda3\lib\site-packages\torch\tensor.py in __array__(self, dtype)
        410     def __array__(self, dtype=None):
        411         if dtype is None:
    --> 412             return self.numpy()
        413         else:
        414             return self.numpy().astype(dtype, copy=False)


    RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.



```python
#So how do we circumvent this? If we don't need gradient on this, we can use

y = np.power(np.linalg.norm(x.detach().numpy()), 2) + 4

```


```python
#But if we do need gradients, we will need to use Pytorch functions instead
y = torch.norm(x).pow(2) + 4
```


```python
y.requires_grad #Since we computer y using x, this also requires grad now
```




    True




```python
#y was created as a result of an operation, so it has a grad_fn unlike x which was created by us

print(x.grad_fn)
print(y.grad_fn)
```

    None
    <AddBackward0 object at 0x00000235754CAA58>



```python
z = y.pow(3).mean()
out = z + 0
print(y)
print(out)
```

    tensor(34., grad_fn=<AddBackward0>)
    tensor(39304., grad_fn=<AddBackward0>)



```python
#Let's backpropagate. Because out is a scalar, we can directly do this
out.backward()
```


```python
print(y.grad)
print(x.grad)
```

    None
    tensor([[ 6936., 13872.],
            [20808., 27744.]])



```python
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
```

    tensor([ -810.8762, -1189.0673,  -784.6422], grad_fn=<MulBackward0>)



```python
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
```


```python
print(x.grad)
```

    tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])



```python
#Another way to differentiate

x = torch.tensor(2., requires_grad=True)

def u(x):
    return x**2
def g(u):
    return -u

dgdx = torch.autograd.grad(g(u(x)), x)
print(dgdx)


y = torch.tensor(2., requires_grad=True)

def u_xy(x, y):
    return x**2 + y**3 + 2*y
def g_xy(u):
    return -u

dgdx = torch.autograd.grad(g_xy(u_xy(x, y)), x)
dgdy = torch.autograd.grad(g_xy(u_xy(x, y)), y)
print(dgdx)
print(dgdy)
```

    (tensor(-4.),)
    (tensor(-4.),)
    (tensor(-14.),)


## Finding Optimal solution using iterative method


```python
w = torch.tensor(torch.randn([3,1]), requires_grad=True) #requires_grad is True as it's a parameter which user initialised

opt = torch.optim.Adam([w], lr=0.1)

def model(x):
    #basis fn
    f = torch.stack([x*x, x, torch.ones(x.shape)], 1)
    #print(f.shape)
    yhat = torch.squeeze(f@w, 1) #why torch.squeeze?
    return yhat

def loss(y, yhat):
    l = torch.nn.functional.mse_loss(yhat, y).sum()
    return l

def generate_data():
    x = torch.rand(100)*20 -10
    y = 5*x*x + 3
    return x, y

def train_step():
    x,y = generate_data()
    yhat = model(x)
    l = loss(y, yhat)

    opt.zero_grad()
    l.backward()
    opt.step()

for _ in range(1000):
    train_step()

print(w.detach().numpy())


```

    C:\Users\hp\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      """Entry point for launching an IPython kernel.


    [[4.9837837e+00]
     [2.8237203e-04]
     [3.9645624e+00]]


We use torch.squeeze to make yhat [100] to avoid broadcasting and getting wrong results. torch.squeeze removes dimensions of 1

C:\Users\hp\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  """Entry point for launching an IPython kernel.
C:\Users\hp\Anaconda3\lib\site-packages\ipykernel_launcher.py:14: UserWarning: Using a target size (torch.Size([100])) that is different to the input size (torch.Size([100, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.

[[ 1.6391281 ]
 [-0.15200512]
 [77.61232   ]]

## What is Broadcasting and why we need to be careful about it?

When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when

they are equal, or
one of them is 1

If these conditions are not met, a ValueError: operands could not be broadcast together exception is thrown, indicating that the arrays have incompatible shapes. The size of the resulting array is the maximum size along each dimension of the input arrays.

>>> x = np.arange(4)
>>> xx = x.reshape(4,1)
>>> y = np.ones(5)
>>> z = np.ones((3,4))

>>> x.shape
(4,)

>>> y.shape
(5,)

>>> x + y
ValueError: operands could not be broadcast together with shapes (4,) (5,)

>>> xx.shape
(4, 1)

>>> y.shape
(5,)

>>> (xx + y).shape
(4, 5)

>>> xx + y
array([[ 1.,  1.,  1.,  1.,  1.],
       [ 2.,  2.,  2.,  2.,  2.],
       [ 3.,  3.,  3.,  3.,  3.],
       [ 4.,  4.,  4.,  4.,  4.]])

>>> x.shape
(4,)

>>> z.shape
(3, 4)

>>> (x + z).shape
(3, 4)

>>> x + z
array([[ 1.,  2.,  3.,  4.],
       [ 1.,  2.,  3.,  4.],
       [ 1.,  2.,  3.,  4.]])


```python
a = torch.rand([5, 3, 5])
b = torch.rand([5, 1, 6])

linear1 = torch.nn.Linear(5, 10)
linear2 = torch.nn.Linear(6, 10)

pa = linear1(a)
print(pa.shape)
pb = linear2(b)
print(pb.shape)
d = torch.nn.functional.relu(pa + pb)

print(d.shape)


```

    torch.Size([5, 3, 10])
    torch.Size([5, 1, 10])
    torch.Size([5, 3, 10])


### A general rule of thumb is to always specify the dimensions in reduction operations and when using torch.squeeze

## Feedforward Neural Network on a Dataset

Steps: (Taken from Ritchie Ng's course)

Step 1. Load Dataset

Step 2. Make Dataset Iterable

Step 3. Create Model Class

Step 4. Instantiate Model Class

Step 5. Instantiate Loss Class

Step 6. Instantiate Optimizer Class

Step 7. Train Model

Step 8. Test Model

To train model (Step 7), we need to follow these steps:

Step 7.1. Convert inputs to tensors with grad accumulation capabilities

Step 7.2. Clear Gradient buffers

Step 7.3. Get output given inputs

Step 7.4. Get loss

Step 7.5. Get gradients w.r.t. parameters

Step 7.6 Update params using gradients

Repeat

We are working on the IRIS dataset and we'll classify it using a standard feed-forward network.
## Loading Data and Making it iterable

References

https://pytorch.org/docs/stable/data.html

https://debuggercafe.com/custom-dataset-and-dataloader-in-pytorch/

https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

First we need to use the Dataset class to instantiate our Dataset. We can either do this using Map-styled dataset (Dataset class) or iterable-style datasets (IterableDataset class). We use Iterable style generally for realtime data and when data is too large for random reads. Here let's try using map-style dataset.



```python
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
from sklearn.preprocessing import LabelEncoder
```


```python
iris = pd.read_csv('iris.csv', header='infer')
```


```python
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
le = LabelEncoder()
le.fit(iris['species'])
iris['species'] = le.transform(iris['species'])
```


```python
iris.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Let's stop for a moment and think what we want. We need to be able to get x, y from our data and for that we inherit from a built-in Pytorch class - Dataset


```python
#Now we need train and test data. But we do need to split the data in a way it's shuffled
from sklearn.model_selection import train_test_split
train, test = train_test_split(iris, test_size=0.1, random_state=5)
```


```python
train.head(20) #now it's shuffled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>5.1</td>
      <td>3.3</td>
      <td>1.7</td>
      <td>0.5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>123</th>
      <td>6.3</td>
      <td>2.7</td>
      <td>4.9</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>130</th>
      <td>7.4</td>
      <td>2.8</td>
      <td>6.1</td>
      <td>1.9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>5.1</td>
      <td>3.7</td>
      <td>1.5</td>
      <td>0.4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4.8</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>71</th>
      <td>6.1</td>
      <td>2.8</td>
      <td>4.0</td>
      <td>1.3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>128</th>
      <td>6.4</td>
      <td>2.8</td>
      <td>5.6</td>
      <td>2.1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>48</th>
      <td>5.3</td>
      <td>3.7</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>4.9</td>
      <td>1.5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>88</th>
      <td>5.6</td>
      <td>3.0</td>
      <td>4.1</td>
      <td>1.3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>74</th>
      <td>6.4</td>
      <td>2.9</td>
      <td>4.3</td>
      <td>1.3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>5.7</td>
      <td>2.9</td>
      <td>4.2</td>
      <td>1.3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>63</th>
      <td>6.1</td>
      <td>2.9</td>
      <td>4.7</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>132</th>
      <td>6.4</td>
      <td>2.8</td>
      <td>5.6</td>
      <td>2.2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>39</th>
      <td>5.1</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>5.5</td>
      <td>2.3</td>
      <td>4.0</td>
      <td>1.3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>5.7</td>
      <td>2.6</td>
      <td>3.5</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5.4</td>
      <td>3.7</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>7.0</td>
      <td>3.2</td>
      <td>4.7</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
class IrisDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        #self.iris_df = pd.read_csv(csv_file)
        self.X = inputs
        self.Y = labels
        self.transform = transform

    def __len__(self):
        #print(len(self.X))
        return len(self.X)

    def __getitem__(self, i):
        data = self.X.iloc[i, :]

        if self.tansform:
            data = self.transform(data)

        if self.Y is not None:
            return (data, self.Y[i])
        else:
            return data

```


```python
train_data = IrisDataset(train.iloc[:, 0:4], train.iloc[:, 4])
test_data = IrisDataset(test.iloc[:, 0:4], test.iloc[:, 4])
```


```python
print(test_data.X)
```

         sepal_length  sepal_width  petal_length  petal_width
    82            5.8          2.7           3.9          1.2
    134           6.1          2.6           5.6          1.4
    114           5.8          2.8           5.1          2.4
    42            4.4          3.2           1.3          0.2
    109           7.2          3.6           6.1          2.5
    57            4.9          2.4           3.3          1.0
    1             4.9          3.0           1.4          0.2
    70            5.9          3.2           4.8          1.8
    25            5.0          3.0           1.6          0.2
    84            5.4          3.0           4.5          1.5
    66            5.6          3.0           4.5          1.5
    133           6.3          2.8           5.1          1.5
    102           7.1          3.0           5.9          2.1
    107           7.3          2.9           6.3          1.8
    26            5.0          3.4           1.6          0.4


Now we have our dataset, let's make it iterable using dataloaders.

We have 135 training examples. We use batch training - first because we don't want to update weights only after going through all the data, and second (not applicable here), if dataset is very large, no batch training might have to load all inputs at once and we might not have that much RAM.

See this thread : https://discuss.pytorch.org/t/i-run-out-of-memory-after-a-certain-amount-of-batches-when-training-a-resnet18/1911/7

Make sure to clean your data structures after each iteration otherwise your program will run out of memory.

Now, if our batch size=5, then 145/5 = 29 iterations

An epoch means we have gone over our whole dataset once. One epoch has 29 iterations in this case.
Let's say we want 10 epochs.

So 10x29 = 290 total iterations.


```python
batch_size = 5
n_iters = 290

num_epochs = n_iters/(len(train_data)/batch_size)

num_epochs = int(num_epochs)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
```

## Let's Create, Instantiate our Model Now and set up Optimizer and Loss

References:

https://github.com/vahidk/EffectivePyTorch

https://pytorch.org/docs/stable/nn.html



First a bit about Pytorch's Modules. According to Pytorch documentation, torch.nn.Module is a Container(fancy name for a class here, but could have been a data structure as well) which acts as a Base class for all Neural Network modules. If we are making a custom neural net, we should subclass it.

"Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes."

When you subclass the Module class, define _ _init_ _() and forward() functions. This allows Pytorch to set up the computational graph and then easily call backward and find gradients.

Side note: "Parameters are essentially tensors with requires_grad set to true. It's convenient to use parameters because you can simply retrieve them all with module's parameters() method"


```python
class FeedForwardNeuralNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.sigmoid = torch.nn.Sigmoid() #This is sigmoid layer, not nn.Functional.sigmoid - that's different
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)

        return out
```


```python
input_dim = 4 #dimension of x, or number of features
hidden_dim = 20
output_dim = 3 #number of classes
```


```python
#Instantiate our model

model = FeedForwardNeuralNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
```

Now let's decide on a loss criterion. Since it's a classification problem, we can use cross-entropy to compute the loss between our model's output softmax distribution and the labels.

The Cross Entropy Function does 2 things at the same time.


1. Computes softmax (logistic/softmax function)
2. Computes cross entropy

See this: https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss


```python
criterion = torch.nn.CrossEntropyLoss()
```


```python

```

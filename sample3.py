import numpy as np
import tensor
import time
import math

np.random.seed(1234567)

def np2dim_idx_val(x):
  return x.shape, np.argwhere(x != 0), x[x.nonzero()]

N = 1000
M = 500
P = 700
EPOCH = 100
threshold = 1

x = np.random.normal(size=[N,M])
x[np.abs(x) < threshold] = 0
y = np.random.uniform(low=-1.0,high=1.0,size=[M,P]) / math.sqrt(M)
z = np.random.uniform(low=-1.0,high=1.0,size=[P,1]) / math.sqrt(P)
label = np.random.uniform(size=[N])

xdims,xidx,xval = np2dim_idx_val(x)
ydims,yidx,yval = np2dim_idx_val(y)
zdims,zidx,zval = np2dim_idx_val(z)
ldims,lidx,lval = np2dim_idx_val(label)
learning_rate = 0.01

start = time.time()

tensor_x = hw3tensor.Tensor(xdims,xidx,xval)
tensor_y = hw3tensor.Tensor(ydims,yidx,yval)
tensor_z = hw3tensor.Tensor(zdims,zidx,zval)
tensor_label = hw3tensor.Tensor(ldims,lidx,lval)

for i in range(EPOCH):
  h1 = tensor_x.matmul(tensor_y) # N*P
  h2 = h1.relu() # N*P
  out = h2.matmul(tensor_z) # N*1
  grad_out = out.reshape([N]).subtract(tensor_label).mult(1.0 / N) # N
  grad_tensor_z = h2.transpose().matmul(grad_out.reshape([N,1])) # P*1
  grad_h2 = grad_out.reshape([N,1]).matmul(tensor_z.transpose())
  grad_h1 = h1.binarilize().elementwise_mult(grad_h2)
  grad_tensor_y = tensor_x.transpose().matmul(grad_h1)

  tensor_y = tensor_y.subtract(grad_tensor_y.mult(learning_rate))
  tensor_z = tensor_z.subtract(grad_tensor_z.mult(learning_rate))
out.print()

end = time.time()
print(str(end - start) + ' seconds')


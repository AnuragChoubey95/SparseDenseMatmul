import numpy as np
import hw3tensor
import time

np.random.seed(1234567)

def np2dim_idx_val(x):
  return x.shape, np.argwhere(x != 0), x[x.nonzero()]

N = 300
M = 5000
P = 1000
threshold = 1

x = np.random.normal(size=[N,M])
x[np.abs(x) < threshold] = 0
y = np.random.normal(size=[M,P])
y[np.abs(y) < threshold] = 0
z = np.random.normal(size=[P,1])
z[np.abs(z) < threshold] = 0

xdims,xidx,xval = np2dim_idx_val(x)
ydims,yidx,yval = np2dim_idx_val(y)
zdims,zidx,zval = np2dim_idx_val(z)

start = time.time()

tensor_x = hw3tensor.Tensor(xdims,xidx,xval)
tensor_y = hw3tensor.Tensor(ydims,yidx,yval)
tensor_z = hw3tensor.Tensor(zdims,zidx,zval)
res = tensor_x.matmul(tensor_y).relu().matmul(tensor_z)
res.print()

end = time.time()
print(str(end - start) + ' seconds')


import mobula
mobula.op.load('MulElemWise')

import mxnet as mx
a = mx.nd.array([1, 2, 3])
b = mx.nd.array([4, 5, 6])
out = mx.nd.empty(a.shape)
mobula.func.mul_elemwise(a.size, a, b, out)
print(out)  # [4, 10, 18]

#cupy version
import cupy as cp
with cp.cuda.Device(0):
    a = cp.array([1,2,3],dtype='float32')
    b = cp.array([4,5,6],dtype='float32')
    out = cp.empty(a.shape,dtype='float32')
    mobula.func.mul_elemwise(a.size, a, b, out)
    print(out)
from numpy import double, float64
import torch

x = torch.empty(3)
y = torch.empty(2,3)
z = torch.rand(2,3)
a = torch.empty(2,3)
b = torch.ones(2,3)
c = torch.ones(2,4, dtype=torch.int)
d = torch.empty(2,3,dtype=torch.double)
e = torch.empty(2,4,dtype=torch.float64)
#Another way of creating a tensor
f = torch.tensor([1.2,3.3,1.9])

print(x,y,z,a,b,c,e,f)
print(c.dtype)
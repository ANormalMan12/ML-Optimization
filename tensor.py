import torch
import torchvision

y=x=torch.tensor([[0,0,1,5,3],
                [1,1,1,2,7],
                [0,3,0,1,34],
                [5,2,4,1,4]])
print(x)

x[0][0]=13 #variable
print(x)

x[:,2]=-5
print(x)

x[2,:]=4
print(x)

x[2,2:4]=-11
print(x)

x[1:,3]=-2
print("x=\n",x)

print("y=\n",y);

print("multiply every element\n",torch.multiply(x,y))
print("equal reloading x*y.T",x*y)
print("matrix multiply",torch.matmul(x,y.T))
print("equal reloading x@y.T",x@y.T)


x.add_(3)
print("possible dangerous: add 3 to all elements\n",x)


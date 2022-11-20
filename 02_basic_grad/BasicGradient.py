import torch

def stopGrad():
    x=torch.tensor([1.,3.0,9  ,5.0][5.,2.3,4.3,2.3],requires_grad=True)
    print(x.requires_grad)
    print((x ** 2).requires_grad)
    with torch.no_grad():
        print((x ** 2).requires_grad)

def prExam():
    x=torch.tensor([1.,3.0, 9,5.0],requires_grad=True)
    y=torch.tensor([2 ,5.1,10,3.4],requires_grad=True)

    z=x+y
    print(z)
    print(z.grad_fn)# where you can find address of x and y

    s=z.sum()
    print(s)
    print(s.grad_fn)

    print("When backward hasn't happened")
    print("x grad",x.grad)
    print("y grad",y.grad)
    print("z grad",z.grad)
    print("s grad",s.grad)

    s.backward();
    print("After s.backward()")
    print("x grad",x.grad)
    print("y grad",y.grad)
    print("z grad",z.grad)
    print("s grad",s.grad)

    # in fact s=(x1+y1)+(x2+y2)+(x3+y3)+(x4+y4)
    # ds/dx=[1,1,1,1]
    # ds/dy=[1,1,1,1]

stopGrad()

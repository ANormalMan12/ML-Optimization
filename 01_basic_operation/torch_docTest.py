import torch

def prMat(str,x):
    print(str,"=",x)
    if(torch.is_tensor(x)):
        print("is tensor")
        print(str,"'s size is",x.numel())
        if(torch.is_storage(x)):
            print("is torch storage")
        else:
            print("isn't torch storage")
    else:
        print("isn't tensor")
    print('\n')

def testTorch():
    mat=torch.ones(5,5)
    mat[1:3]=5#from Row 1 to Row 2
    pyarr=[[3,3,12,12,4],[4,12,4,6,7]]
    torArr=torch.tensor(pyarr)
    prMat("mat",mat)
    prMat("pyar",pyarr)
    prMat("toArr",torArr)
    print("Examples show that tensor is not equal to storage")

def testMatSliceCat():
    mat=torch.randn(5,3)
    print(mat)
    print(mat[0])#get the first row
    rAddMat=torch.randn(2,3)
    rowComMat=torch.cat([mat,rAddMat])#row combination
    prMat("row:",rowComMat)# 7*3 matrix

    cAddMat=torch.randn(5,11)
    colComMat=torch.cat([mat,cAddMat],1)#col combination. 1
    prMat("col:",colComMat)# 5*14 matrix
def testReshape():
    print('\n')
    A=torch.randn(5,4)
    print(A)
    A2=A.reshape(2,10)
    A3=A.view(10,2)
    print(A2)
    print(A3)
    print("Remember: A itself will not be changed by (reshape,view) function")

#testMatSliceCat()
#testTorch()
testReshape()


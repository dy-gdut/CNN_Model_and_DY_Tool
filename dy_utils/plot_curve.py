from matplotlib import pyplot as plt
import torch
import numpy
import torch.nn as nn

def plot_curve(data,hook):
    fig=plt.figure()
    plt.plot(range(len(data)),data,color='blue')
    # plt.legend(['value'],loc='upper right')
    plt.legend(['{}'.format(hook)], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('{}'.format(hook))
    plt.savefig('{}.jpg'.format(hook))
    plt.show()

def plot_image(img,label,name):
    fig=plt.figure()
    dict = {
        0: 'Green Dragon',
        1: 'Fire Dragon',
        2: 'Purple Dragon',
        3: 'Pikaqiu',
        4: 'Squirtle'
    }
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        # img=img.numpy()
        # img=img.transpose((2,0,1))
        plt.imshow(img[i][0]*0.3081+0.1307,interpolation='none')
        plt.title("{}:{}".format(name,dict[label[i].item()]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def one_hot(label,depth=10):
    out=torch.zeros(label.size(0),depth)
    idx=torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1,index=idx,value=1)
    return out

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self,x):
        shape=torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1,shape)
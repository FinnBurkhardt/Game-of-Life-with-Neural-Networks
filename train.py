import os

import sys
import torch
from torch import nn
from tqdm import tqdm
from models import CNN, MLP,CNN_residual, MLP_residual
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

def main():

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")





    MODEL='CNN_residual'#Choose Model from 1. MLP
                        #                  2. CNN
                        #                  3. MLP_residual
                        #                  4. CNN_residual


    PLOT_FILTERS=True
    EPOCHS = 50
    BATCH_SIZE=512
    DATADIR_TRAIN = '/home/finn/Desktop/CGOL/data/train'    #Training Data Directory
    DATADIR_VAL = '/home/finn/Desktop/CGOL/data/val'        #Validation Data Directory
    LEARNING_RATE = 0.0001




    fig, axs = plt.subplots(3)


    FramesTrain = []
    nextFramesTrain = []
    FramesVal = []
    nextFramesVal = []


    #Load Training Data in Memory
    for j in range(int(1*len(os.listdir(DATADIR_TRAIN)))):
        for i in range(29):
            FramesTrain.append(np.load(DATADIR_TRAIN+'/'+str(j).zfill(3)+'/'+str(i).zfill(3)+'.npy')/255)
            nextFramesTrain.append(np.load(DATADIR_TRAIN+'/'+str(j).zfill(3)+'/'+str(i+1).zfill(3)+'.npy')/255)




    #Load Validation Data in Memory
    for j in range(int(1*len(os.listdir(DATADIR_VAL)))):
        for i in range(29):
            FramesVal.append(np.load(DATADIR_VAL+'/'+str(j).zfill(3)+'/'+str(i).zfill(3)+'.npy')/255)
            nextFramesVal.append(np.load(DATADIR_VAL+'/'+str(j).zfill(3)+'/'+str(i+1).zfill(3)+'.npy')/255)


    net=net=getattr(sys.modules[__name__], MODEL)().to(dev)#CNN_residual().cuda()


    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()


    for epoch in range(EPOCHS):                                                             #iterate over hole dataset for 'EPOCH'times
        Losses=[] 
        for i in tqdm(range(0, len(FramesTrain), BATCH_SIZE)):             
            X = torch.Tensor(FramesTrain[i:int(i+BATCH_SIZE)]).view(-1,1,16,16).to(dev)#.cuda()       #get and reshape one batch of inputs
            y = torch.Tensor(nextFramesTrain[i:int(i+BATCH_SIZE)]).view(-1,1,16,16).to(dev)#.cuda()   #get and reshape one batch of labels

            net.zero_grad()  
            output = net(X).view(-1,1,16,16)                                            #reshape the output to match image shape (1,16,16)

            loss = loss_function(output, y)                                             #calc loss value
            Losses.append(loss)
            loss.backward()                                                             #apply the loss backwards
            optimizer.step()                                                            #attempt to optimize weights to account for loss/gradients

        Loss = sum(Losses)/len(FramesTrain)
        #print(Loss.item())



        valLosses=[]
        for i in tqdm(range(0, len(FramesVal), BATCH_SIZE)): 
            X = torch.Tensor(FramesTrain[i:int(i+BATCH_SIZE)]).view(-1,1,16,16).to(dev)#.cuda()       #get and reshape one batch of inputs
            y = torch.Tensor(nextFramesTrain[i:int(i+BATCH_SIZE)]).view(-1,1,16,16).to(dev)#.cuda()   #get and reshape one batch of labels

            net.zero_grad()  
            output = net(X).view(-1,1,16,16)                                            #reshape the output to match image shape (1,16,16)

            loss = loss_function(output, y)                                             #calc loss value
            valLosses.append(loss)
              
        valLoss = sum(valLosses)/len(FramesVal)
        #print(valLoss.item())
        
        a=X.detach().cpu().numpy()                  #prepair label (Frame) to be ploted
        b=y.detach().cpu().numpy()                  #prepair label (nextFrame) to be ploted
        c=output.detach().cpu().numpy()             #prepair output to be ploted


        #prepair and plot results
        axs[0].axis('off')
        axs[0].set_title('state at time t')
        axs[0].imshow(a[0].reshape((16,16)))
        axs[1].axis('off')
        axs[1].set_title('state at time t+1')
        axs[1].imshow(b[0].reshape((16,16)))
        axs[2].axis('off')
        axs[2].set_title('reconstructed state at time t+1')
        axs[2].imshow(c[0].reshape((16,16)))

        lastEpoch = (EPOCHS==(epoch))
        plt.show(block=lastEpoch)                   #block if it is the last epoch
        plt.pause(1)


        print("Epoch: "+str(epoch)+"  | Train Loss: "+str(Loss)+"  | Val Loss: "+str(valLosses))





    #plot filters of CNN
    if (MODEL=='CNN' or MODEL=='CNN_residual') and PLOT_FILTERS==True:
        filter = net.layers[0].weight.data.clone().detach().cpu()#get weight from first Layer and loads them to cpu
        
        #plot filters  
        grid = utils.make_grid(filter, nrow=9, normalize=True, padding=1)
        plt.figure( figsize=(9,1) )
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
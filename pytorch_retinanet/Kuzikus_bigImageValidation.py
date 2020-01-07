"""
    Contains functionality to evaluate a model on a big image.
    The big image is chopped into patches (with a stride),
    the patches are evaluated one by one, and the result is
    then stitched together again.
    
    2017 bkellenb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import TensorSharding
import numpy as np

    

def getOutputSize(model,inputSize):
#     if isinstance(model,nn.DataParallel):
#         depth = next(list(model.module.modules())[1].children()).in_channels
#     else:
#         depth = next(list(model.modules())[1].children()).in_channels
        
    depth = model.getParameters()[0].size()[1]  #TODO
        
    if isinstance(inputSize,int):
        inputSize = [1,depth,inputSize,inputSize]
    elif len(inputSize)==2:
        inputSize = [1,depth,inputSize[0],inputSize[1]]
    elif len(inputSize==3):
        inputSize = [1,inputSize[0],inputSize[1],inputSize[2]]
        
    dummyTensor = torch.Tensor(inputSize[0],inputSize[1],inputSize[2],inputSize[3])
    if next(model.parameters()).is_cuda:
        dummyTensor = dummyTensor.cuda()
    
    with torch.no_grad():
        output = model(dummyTensor)
                   
    if isinstance(output, tuple):
        output = output[0]
        
        
    return list(output.size())



def evalOnBigTensor(model,tensor,shardSize,batchSize,stride='auto',restoreMode='average',doSoftmax=True,exportFeatureVectors=False,feSize=-1):
    """
        Evaluates the model on a big tensor.
        Flags:
        - model: the PyTorch model to use
        - tensor: the big tensor to evaluate
        - shardSize: tile size the tensor should be split into
        - stride: tile stride (max. shardSize)
        - batchSize
        - restoreMode
                         
        Returns:
        - TODO
    """
    sz = tensor.size()
    
    
    if isinstance(shardSize,int) or isinstance(shardSize,float):
        shardSize = [shardSize,shardSize]
    
    if len(shardSize)==1:
        shardSize = [shardSize[0],shardSize[0]]
    shardSize = [int(round(i)) for i in shardSize]
    
    
    # determine output tensor size
    if hasattr(model,'getOutputSize'):
        outSize = model.getOutputSize(np.array(shardSize))
    else:
        outSize = getOutputSize(model,shardSize)
    
    
    if isinstance(stride,str) and stride=='auto':
        stride = shardSize.copy()
    else:
        if isinstance(stride,int) or isinstance(stride,float):
            stride = [stride,stride]
        
        if len(stride)==1:
            stride = [stride[0],stride[0]]
        
        stride = [int(round(i)) for i in stride]
        stride = [max(1,i) for i in stride]
        stride = [min(shardSize[i],stride[i]) for i in range(len(stride))]
        
    
#     gridX,gridY = TensorSharding.createSplitLocations_equalInterval(sz[1:],shardSize,stride,True)
    gridX,gridY = TensorSharding.createSplitLocations_auto(sz[1:],stride,True)
    
    scaleX = float(shardSize[0]) / float(outSize[2])
    scaleY = float(shardSize[1]) / float(outSize[3])
    
    gridX_out = gridX + np.abs(np.min(gridX))
    gridX_out = np.round(gridX_out / scaleX).astype(int)
    
    gridY_out = gridY + np.abs(np.min(gridY))
    gridY_out = np.round(gridY_out / scaleY).astype(int)
    
    
    tensors = TensorSharding.splitTensor(tensor,shardSize,gridX,gridY)
    numPatches = tensors.size()[0]
    
    
    # evaluate
    probs = torch.Tensor(numPatches,outSize[1],outSize[2],outSize[3])
    
    if exportFeatureVectors:
        fVecs = torch.Tensor(numPatches,feSize,outSize[2],outSize[3])
    
    numBatches = int(np.ceil(numPatches / float(batchSize)))
    

    
    for t in range(0,numBatches):
        startIdx = t*batchSize
        endIdx = min((t+1)*batchSize,numPatches)
        
        batch = tensors[startIdx:endIdx,:,:,:]
        
        if len(batch.size())==3:
            batch = batch.unsqueeze(0)
        
        if next(model.parameters()).is_cuda:
            batch = batch.cuda()
        with torch.no_grad():
            classPred,fVec = model(batch)
            if doSoftmax:
                classPred = F.softmax(classPred,dim=1)
            classPred = classPred.view(endIdx-startIdx,outSize[1],outSize[2],outSize[3])
            classPred = classPred.cpu()
            probs[startIdx:endIdx,:,:,:] = classPred

        
            if exportFeatureVectors:
                fVec = fVec.view(endIdx-startIdx,feSize,outSize[2],outSize[3])
                fVec = fVec.cpu()
                fVecs[startIdx:endIdx,:,:,:] = fVec
            
        
    # restore prediction_class
    prediction_class = TensorSharding.combineShards(probs, gridX_out, gridY_out, None, restoreMode)
    
    if exportFeatureVectors:
        prediction_fVec = TensorSharding.combineShards(fVecs, gridX_out, gridY_out, None, restoreMode)
        return prediction_class,prediction_fVec
    else:
        return prediction_class

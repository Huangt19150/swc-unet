# custom libraries
from code_learning import helper_functions

# standard libraries
import torch
import torch.utils.data 
import numpy as np
import matplotlib.pyplot as plt


# Dataset classes
class Projection_DS(torch.utils.data.Dataset):
    def __init__(self, settings, img_paths, label_paths, mode):
        """
        Loads & returns tensors of single projection (and mask) based on paths to patch & projection
        --> this is used for 2D-based training & validation where each projection is treated independently
        """
        self.settings = settings
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.mode = mode

    def __getitem__(self, index):
        data = np.load(self.img_paths[index])
        if(self.label_paths is None):
            # We are in prediction mode -> no mask available.
            img = helper_functions.image_to_tensor(data)
            return img
        else:
            # We are in training/validating/testing mode -> also load & return mask
            mask = np.load(self.label_paths[index])
            if(self.settings['trafo']):
                if(self.settings['trafo'] == 'flip_augm'):
                    img, mask = data, mask
                else:                        
                    raise ValueError('Chosen transformaton does not exist')
            img = helper_functions.image_to_tensor(data)
            mask = helper_functions.array_to_tensor(mask)
            pid = self.label_paths[index].split('/')[-1].split('_')[2]
            if self.mode == 'train':
                return pid, img, mask
            else:
                return self.img_paths[index].split('/')[-1].split('.tif')[0], img, mask

    def __len__(self):
        return len(self.img_paths)
    
class Projection_DS_ske(torch.utils.data.Dataset):
    def __init__(self, settings, img_paths, label_paths, ske_paths, mode):
        """
        Loads & returns tensors of single projection (and mask) based on paths to patch & projection
        --> this is used for 2D-based training & validation where each projection is treated independently
        """
        self.settings = settings
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.ske_paths = ske_paths
        self.mode = mode

    def __getitem__(self, index):
        data = np.load(self.img_paths[index]) 

        if(self.label_paths is None):
            # We are in prediction mode -> no mask available.
            img = helper_functions.image_to_tensor(data)
            return img
        else:
            mask = np.load(self.label_paths[index])                                                     
            if(self.settings['trafo']):
                if(self.settings['trafo'] == 'flip_augm'):
                    img, mask = data, mask
                else:                        
                    raise ValueError('Chosen transformaton does not exist')
            img = helper_functions.image_to_tensor(data)
            mask = helper_functions.array_to_tensor(mask)
            ske = np.load(self.ske_paths[index])                                                     
            ske = helper_functions.array_to_tensor(ske)
    
            pid = self.label_paths[index].split('/')[-1].split('_')[2]
            if self.mode == 'train':
                return pid, img, mask, ske
            else:
                return self.img_paths[index].split('/')[-1].split('.tif')[0], img, mask

    def __len__(self):
        return len(self.img_paths)

class Volumetric_DS(torch.utils.data.Dataset):
    def __init__(self, settings, pids, folderData, folderLabels=None):
        """
        Loads & returns tensors of all 3 projections of input (and mask) based on patch IDs.
        --> This is used for evaluation of test set where we want to assess the full power by
        making use of all 3 projections at once.
        --> this is NOT used for training
        Thus, no augmentation or maskweighting is needed/valid
        """
        self.settings = settings
        self.pids = pids
        self.folderData = folderData
        self.folderLabels = folderLabels

    def __getitem__(self, index):

        pid = self.pids[index]
        
        imgs = torch.tensor(np.zeros((3,242,242),np.float32))#.............................................................
        for d, dim in enumerate(['Y','X','Z']):
            data = np.load(self.folderData + 'data_patch_' + str(pid) + "_" + dim + '.npy')
            imgs[d] = helper_functions.image_to_tensor(data)

        if(self.folderLabels is None):
            # We are in prediction mode -> no masks available.
            return pid, imgs
        else:
            # We are in validating or testing mode -> also load & return masks
            masks = torch.tensor(np.zeros((3,242,242),np.float32))#...........................................................
            for d, dim in enumerate(['Y','X','Z']):
                mask = np.load(self.folderLabels + 'label_patch_' + str(pid) + "_" + dim + '.npy')
                masks[d] = helper_functions.array_to_tensor(mask)
            return pid, imgs, masks

    def __len__(self):
        return len(self.pids)

# Data fetcher functions
def get_paths(pids, settings):
    """Return the path to images and labels given a (list of) patch id"""
    folderData   = 'data/input/input_' + settings['description'] + '/'   
    folderLabels = 'data/labels/labels_' + settings['description'] + '/'  

    imgPaths   = []
    labelPaths = []
    for pid in pids:
        for dim in ['Y','X','Z']:
            imgPaths.append(folderData  + 'data_patch_'  + str(pid) + "_" + dim + ".npy")
            labelPaths.append(folderLabels + 'label_patch_' + str(pid) + "_" + dim + ".npy") 

    return imgPaths, labelPaths

def get_paths_ske(pids, settings):
    """Return the path to images and labels given a (list of) patch id"""
    folderData   = 'data/input/input_' + settings['description'] + '/'   
    folderLabels = 'data/labels/labels_' + settings['description'] + '/'  
    folderSkes = 'data/skeletons/skeletons_' + settings['description'] + '/'  

    imgPaths   = []
    labelPaths = []
    skePaths = []
    for pid in pids:
        for dim in ['Y','X','Z']:
            imgPaths.append(folderData  + 'data_patch_'  + str(pid) + "_" + dim + ".npy")
            labelPaths.append(folderLabels + 'label_patch_' + str(pid) + "_" + dim + ".npy") 
            skePaths.append(folderSkes + 'ske_patch_' + str(pid) + "_" + dim + ".npy") 

    return imgPaths, labelPaths, skePaths


# Data normalization & augmentation
def normalize_patch(data, method):
    if(type(method) is int):     
        img = np.clip(data / method, 0, 1)
#     elif(method == 'globalmax'): 
#         img = data['raw'] / data['global_mval']
    elif(method == 'localmax'):  
        img = data / np.max([np.max(data),0.001])
    elif('localZMZD' in method):  
        Mean = np.mean(data)
        Max = np.max([np.max(data),0.001])
        Devi = np.std(data)**2
        img = (data - Mean)/Devi
#     elif('localmaxZMZD' in method): 
#         img = (data/np.max([np.max(data),0.001]) - 0.217)/0.180
    else:
        raise ValueError('Chosen normalization does not exist')
    return img


# Data loader wrappers
def projection_loader(settings, pids, threads, use_cuda, mode):
    '''
    Returns data loader object for 2D training or validation (pixel-level)
    '''
    imgPaths, labelPaths = get_paths(pids, settings) 
    dataSet = Projection_DS(settings, imgPaths, labelPaths, mode)
    if mode == 'train':
        dataLoader = torch.utils.data.DataLoader(dataSet, int(settings['batchSize']),
                                                 sampler=torch.utils.data.sampler.RandomSampler(dataSet),
                                                 num_workers=threads, pin_memory=use_cuda)
    else:
        dataLoader = torch.utils.data.DataLoader(dataSet, 1,
                                                 sampler=torch.utils.data.sampler.RandomSampler(dataSet),
                                                 num_workers=threads, pin_memory=use_cuda)
        
    return dataLoader

def projection_loader_ske(settings, pids, threads, use_cuda, mode):
    '''
    Returns data loader object for 2D training or validation (pixel-level)
    '''
    imgPaths, labelPaths, skePaths = get_paths_ske(pids, settings) 
    dataSet = Projection_DS_ske(settings, imgPaths, labelPaths, skePaths, mode)
    if mode == 'train':
        dataLoader = torch.utils.data.DataLoader(dataSet, int(settings['batchSize']),
                                                 sampler=torch.utils.data.sampler.RandomSampler(dataSet),
                                                 num_workers=threads, pin_memory=use_cuda)
    else:
        dataLoader = torch.utils.data.DataLoader(dataSet, 1,
                                                 sampler=torch.utils.data.sampler.RandomSampler(dataSet),
                                                 num_workers=threads, pin_memory=use_cuda)
        
    return dataLoader

def volumetric_loader(settings, pids, threads, use_cuda):
    '''
    Returns data loader object for 3D validation or testing (metastasis-level)
    '''
    folderData   = 'data/input/input_' + settings['description'] + '/'   
    folderLabels = 'data/labels/labels_' + settings['description'] + '/'    
    dataSet = Volumetric_DS(settings, pids, folderData, folderLabels)
    dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=1,
                                             sampler=torch.utils.data.sampler.SequentialSampler(dataSet),
                                             num_workers=threads, pin_memory=use_cuda)
    return dataLoader


# custom libraries
from code_learning import helper_functions
from code_learning import architecture

# standard libraries
import os
import torch
import numpy as np
import copy
import time
from tqdm import tqdm
import datetime

# Define model class "Segmenter"
class Segmenter:
    def __init__(self, net, settings):
        """
        A wrapper class for training the network as well as making predictions
        Args:
            net (nn.Module): The neural net module containing the definition of your model
            settings: dictionary with all settings
        """
        # determine ID for this run
        pathtologs = 'model/traininglogs/'+settings['description']
        did = 1
        while(os.path.isdir(pathtologs + '_'+str(did))): did += 1
        # save key parameters to class
        self.use_cuda = torch.cuda.is_available()
        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), settings['learningRate'])
        self.lr_scheduler  = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [200,250], gamma=0.1, last_epoch=-1)
        self.settings = copy.deepcopy(settings) # settings decoupled from input
        self.description = self.settings['description']#+'_'+str(did)
        self.epoch_counter = 0
        self.classweights = None # needed as shifted to GPU

    def restore_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path, map_location='cpu'))

    def compute_loss(self, logits, labels):
        if(self.classweights is None):
            self.classweights = torch.tensor(self.settings['BCEWeights'])

            if self.use_cuda:
                self.classweights = self.classweights.cuda()
        l = helper_functions.WeightedBCELoss2d().forward(logits, labels, self.classweights)
        return l
    
    def compute_loss_dynamicBCE(self, logits, labels):
        if(self.classweights is None):
            pos = 1.*torch.sum(labels ==1)
            neg = 1.*torch.sum(labels ==0)
            N = pos + neg
            self.classweights = torch.tensor([[N/pos,N/neg]]).cuda()
            if self.use_cuda:
                self.classweights = self.classweights.cuda()
        l = helper_functions.WeightedBCELoss2d().forward(logits, labels, self.classweights)
        return l
    
    def compute_loss_ske(self, logits, labels, skes):
        ske_weight = 5. #20. #10. #5. #...............................................................adjust!
        if(self.classweights is None):
            self.classweights = torch.tensor(self.settings['BCEWeights'])
            if self.use_cuda:
                self.classweights = self.classweights.cuda()
        l = helper_functions.WeightedBCELoss2d_ske().forward(logits, labels, skes, self.classweights, ske_weight)
        return l

    def train_epoch_ske(self, train_loader):
        losses = helper_functions.AverageMeter()
        precisions = helper_functions.AverageMeter()
        recalls = helper_functions.AverageMeter()
        f1s = helper_functions.AverageMeter()        
        
        batch_size = train_loader.batch_size # variable in DataLoader class as inherited by torch.utils.data.DataLoader
        it_count = len(train_loader)
        with tqdm(total=it_count, desc="Training..", bar_format='{l_bar}{bar}| Batch {n_fmt}/{total_fmt}') as pbar:
            for ind, (pid, data, mask, ske) in enumerate(train_loader):
                # Define autograd variables
                if self.use_cuda: # equivalent to if(torch.cuda.is_available())
                    data = data.cuda()
                    mask = mask.cuda()
                    ske = ske.cuda()
                data = torch.autograd.Variable(data)
                mask = torch.autograd.Variable(mask,requires_grad=False)
                ske = torch.autograd.Variable(ske,requires_grad=False)
                # forward pass
                logits = self.net.forward(data)
                probs = torch.nn.functional.sigmoid(logits)
                pred = (probs > self.settings['threshold']).float()
                # backward pass for gradients + optimization step
                loss = self.compute_loss_ske(logits, mask, ske)
                self.optimizer.zero_grad() # clears the gradients
                loss.backward() 
                self.optimizer.step() # updates the weights based on the gradients
                
                # update performance statistics
                losses.update(loss.item(), batch_size) 
                del loss
                precision, recall, f1 = helper_functions.compute_prf(pred, mask)
                precisions.update(precision)
                recalls.update(recall)
                f1s.update(f1)
                del precision, recall, f1

                # update pbar
                pbar.update(1)

        return losses.avg, precisions.avg, recalls.avg, f1s.avg

    def train_ske(self, train_loader, settings): # valid_loader
        """
            Trains the neural net
        Args:
            train_loader (DataLoader): The Dataloader for training
            valid_loader (DataLoader): The Dataloader for validation
        Returns:
            str, None: The path where the model was saved, or None if it wasn't saved
        """
        now = datetime.datetime.now()
        if self.use_cuda:
            self.net.cuda()
#         log = open('./train_log.txt', 'w')
        while(self.epoch_counter+1 <= self.settings['epochs']):
            tqdm.write("Running epoch {}/{}".format(self.epoch_counter + 1, self.settings['epochs']))
            time.sleep(0.5)
            # train epoch
            self.net.train()
            train_loss_and_prf = self.train_epoch_ske(train_loader)
            self.lr_scheduler.step() 
            print(train_loss_and_prf)
#             log.write(str(train_loss_and_prf) + '\n')
            time.sleep(0.5)
            # Update counter for next iteration        
            self.epoch_counter += 1
            if self.epoch_counter % 300 == 0:
                PATH = './model/' + 'day_' + str(now.day)+ '_hour_' + str(now.hour)+'_min_' + str(now.minute)+'_model_saved_after_' + str(self.epoch_counter) + '_epochs.pkl'
                torch.save(self.net.state_dict(), PATH)
        return

    def train_epoch(self, train_loader):
        losses = helper_functions.AverageMeter()
        precisions = helper_functions.AverageMeter()
        recalls = helper_functions.AverageMeter()
        f1s = helper_functions.AverageMeter()        
        
        batch_size = train_loader.batch_size # variable in DataLoader class as inherited by torch.utils.data.DataLoader
        it_count = len(train_loader)
        with tqdm(total=it_count, desc="Training..", bar_format='{l_bar}{bar}| Batch {n_fmt}/{total_fmt}') as pbar:
            for ind, (pid, data, mask) in enumerate(train_loader):
                # Define autograd variables
                if self.use_cuda: # equivalent to if(torch.cuda.is_available())
                    data = data.cuda()
                    mask = mask.cuda()
                data = torch.autograd.Variable(data)
                mask = torch.autograd.Variable(mask,requires_grad=False)
                # forward pass
                logits = self.net.forward(data)
#                 print('logit:',logits.size())
#                 print('mask:',mask.size())          
                probs = torch.nn.functional.sigmoid(logits)
                pred = (probs > self.settings['threshold']).float()
                # backward pass for gradients + optimization step
                loss = self.compute_loss(logits, mask)
                self.optimizer.zero_grad() # clears the gradients
                loss.backward() 
                self.optimizer.step() # updates the weights based on the gradients
                
                # update performance statistics
                losses.update(loss.item(), batch_size) 
                del loss
                precision, recall, f1 = helper_functions.compute_prf(pred, mask)
                precisions.update(precision)
                recalls.update(recall)
                f1s.update(f1)
                del precision, recall, f1

                # update pbar
                pbar.update(1)

        return losses.avg, precisions.avg, recalls.avg, f1s.avg

    def train(self, train_loader, settings): # valid_loader
        """
            Trains the neural net
        Args:
            train_loader (DataLoader): The Dataloader for training
            valid_loader (DataLoader): The Dataloader for validation
        Returns:
            str, None: The path where the model was saved, or None if it wasn't saved
        """
        now = datetime.datetime.now()
        if self.use_cuda:
            self.net.cuda()
        while(self.epoch_counter+1 <= self.settings['epochs']):
            tqdm.write("Running epoch {}/{}".format(self.epoch_counter + 1, self.settings['epochs']))
            time.sleep(0.5)
            # train epoch
            self.net.train()
            train_loss_and_prf = self.train_epoch(train_loader)
            self.lr_scheduler.step() 
            print(train_loss_and_prf)
#             log.write(str(train_loss_and_prf) + '\n')
            time.sleep(0.5)
            # Update counter for next iteration        
            self.epoch_counter += 1
            if self.epoch_counter % 300 == 0:
                PATH = './model/' + 'day_' + str(now.day)+ '_hour_' + str(now.hour)+'_min_' + str(now.minute)+'_model_saved_after_' + str(self.epoch_counter) + '_epochs.pkl'
                torch.save(self.net.state_dict(), PATH)
#         log.close()
        return
            
    def test_2d(self, test_loader, settings, savePath=None):
        self.net.eval() # Switch to evaluation mode
        losses = helper_functions.AverageMeter()
        precisions = helper_functions.AverageMeter()
        recalls = helper_functions.AverageMeter()
        f1s = helper_functions.AverageMeter()  
        with tqdm(total=len(test_loader), desc="Evaluation", bar_format='{l_bar}{bar}| Patch {n_fmt}/{total_fmt}') as pbar:
            for ind, (name, img, mask) in enumerate(test_loader):
                mask = mask.squeeze()

                # Move to GPU & turn off AutoGrad
                if self.use_cuda:
                    self.net = self.net.cuda()
                    img = img.cuda()
                with torch.no_grad():
                    img = torch.autograd.Variable(img)
                    logit = self.net.forward(img)
                    prob = torch.nn.functional.sigmoid(logit)
                    pred = (prob > settings['threshold']).float()
                    # Save prob/pred
                    if savePath:
                         np.save(savePath + name[0][11:-4] + '.npy', prob.cpu().numpy().squeeze()) #.... -for generate secondary input
                    		
                    # cal. p,r,f
                    precision, recall, f1 = helper_functions.compute_prf(pred, mask)
                    precisions.update(precision)
                    recalls.update(recall)
                    f1s.update(f1)
                    del precision, recall, f1

		            # cal. loss
                    loss = self.compute_loss(logit.cpu(), mask)
                    losses.update(loss.item()) 
                    del loss
          
                pbar.update(1)
        print('2d loss, precision, recall, f1 scores: ' + str(losses.avg),str(precisions.avg),str(recalls.avg),str(f1s.avg))
        return

    def test_3d(self, data_loader, settings, mode, savePath=None,THR_dim=200,): 
        '''
        '''
        labelPath_3d = '/home/guolab/pythonScript/data/labels_3d/labels_3d_' + settings['description'] + '/'
        print('mode: ' + mode + ', threshold: ' + str(settings['threshold']))        
        self.net.eval() # Switch to evaluation mode
        probsvol_collect = {}
        precisions = helper_functions.AverageMeter()
        recalls = helper_functions.AverageMeter()
        f1s = helper_functions.AverageMeter()  
        
        with tqdm(total=len(data_loader), desc="Evaluation", bar_format='{l_bar}{bar}| Patch {n_fmt}/{total_fmt}') as pbar:
            for ind, (pid, imgs, masks) in enumerate(data_loader):
                evaluation = {}
                evaluation['pid'] = int(pid.numpy().squeeze())
                evaluation['imgs'] = imgs.numpy().squeeze()
                evaluation['masks_3d_sub'] = np.load(labelPath_3d + 'label_patch_' + str(pid.numpy()[0]) + '.npy')
                
                probmaps = torch.tensor(np.zeros((3,242,242),np.float32))
                logitsmaps = torch.tensor(np.zeros((3,242,242),np.float32))
                # Move to GPU & turn off AutoGrad
                if self.use_cuda:
                    self.net = self.net.cuda()
                    imgs = imgs.cuda()
                    probmaps = probmaps.cuda()
                with torch.no_grad():
                    imgs = torch.autograd.Variable(imgs)
                    probmaps = torch.autograd.Variable(probmaps)
                # Predict 3 projections
                for d, dim in enumerate(['Y','X','Z']):
                    # forward pass
                    img = imgs[:,d,:,:].unsqueeze(1) # (channel, dimension, height, width)
                    logits = self.net(img)
                    probs = torch.sigmoid(logits)
                    probmaps[d,:,:] = probs.detach()
                    logitsmaps[d,:,:] = logits.detach()

                # Get outputs back to CPU & turn into 3D prediction
                logits = logitsmaps.cpu()
                probs = probmaps.cpu()
                evaluation['logits'] = logits.numpy().squeeze()
                evaluation['probs'] = probs.numpy().squeeze()
                probsvol = helper_functions.backproject_probabilities((242,242,242), mode, Yproj=probs[0].numpy(), Xproj=probs[1].numpy(), Zproj=probs[2].numpy())#.............................................................................
                
                # Save probsvol
                if savePath:
                    np.save(savePath + str(pid.numpy()[0]) + '.npy',probsvol)
                
                # cal. p,r,f for 3d sub-blocks
                predsvol = (probsvol > settings['threshold']) 
                precision, recall, f1 = helper_functions.compute_prf(predsvol, evaluation['masks_3d_sub'], store = 'cpu')
                precisions.update(precision)
                recalls.update(recall)
                f1s.update(f1)
                del precision, recall, f1
                
                probsvol_collect[str(ind)] = probsvol
                pbar.update(1)
                
        print('3d precision, recall, f1 scores: ' + str(precisions.avg),str(recalls.avg),str(f1s.avg))
                
        return probsvol_collect
        
        
def select_architecture(settings):
        if(settings['architecture'] == 'UNet768'):
              return architecture.UNet768()
        elif(settings['architecture'] == 'UNet768_ch32'):
              return architecture.UNet768_ch32()
        else:
            print('Architecture not specified')
            return None

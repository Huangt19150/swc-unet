# standard libraries
import torch
import numpy as np
import pickle

# Transformation functions
def image_to_tensor_ch8(image, mean=0, std=1.):
    image = image.astype(np.float32)
    tensor = torch.from_numpy(image)
    return tensor

def image_to_tensor(image, mean=0, std=1.):
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)
    tensor = torch.from_numpy(image)
    return tensor

def array_to_tensor(array):
    array = array.astype(np.float32)
    tensor = torch.from_numpy(array).type(torch.FloatTensor)
    return tensor

def backproject_probabilities(shape, mode, Yproj=None, Xproj=None, Zproj=None):
    '''
    volume = backproject_probabilities(shape, Yproj=None, Xproj=None, Zproj=None, mode='multiply')

    Takes 1-3 two-dimensional probability maps and backprojects them to a common
    volumetric representation. You can choose between 3 different modes:
        - multiply: probabilities are multiplied
        - average: probabilities are averaged
        - voting: like multiplication, but will return 100% for voxels where at least 2 of 3 votes are very strong

    Multiply is very clean but will lead to false negatives in cases where one probability is 0 (for some reason)
    even if we have a strong believe from the other two probabilities. Average is also clean, but will lead to
    false positives in dense plots (example: any 100% blob will cause a cross-like corridor where any further 50%
    probability will pass the hurdle even if the third probability is 0%). Voting combines the best of both.
    '''
    yvol = np.zeros(shape) if (Yproj is None) else np.moveaxis(np.asarray([Yproj for y in range(0, shape[0])]), 0, 0)
    xvol = np.zeros(shape) if (Xproj is None) else np.moveaxis(np.asarray([Xproj for x in range(0, shape[1])]), 0, 1)
    zvol = np.zeros(shape) if (Zproj is None) else np.moveaxis(np.asarray([Zproj for Z in range(0, shape[2])]), 0, 2)
    if (mode == 'average'):
        return (yvol + xvol + zvol) / 3
    elif (mode == 'voting'):
        vote1 = yvol * xvol * zvol  # accepted if accepted for multiplication
        vote2 = ((yvol + xvol + zvol) / 3 > 0.67)  # accept if 67% sure on average
        vote3a = (yvol * xvol > 0.9 ** 2)  # accept if 2x >90%
        vote3b = (yvol * zvol > 0.9 ** 2)  # accept if 2x >90%
        vote3c = (zvol * xvol > 0.9 ** 2)  # accept if 2x >90%
        return np.clip((vote1 + vote2 + vote3a + vote3b + vote3c), 0, 1)
    else:
        return (yvol * xvol * zvol)

# Loss
class WeightedBCELoss2d(torch.nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, targets, classweights):
        probs = torch.nn.functional.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        loss = classweights[0][0] * (targets_flat * torch.log(probs_flat + 0.0001)) + classweights[0][1] * (
                    (1 - targets_flat) * torch.log(1 - probs_flat + 0.0001))
        return torch.neg(loss.sum())

class WeightedBCELoss2d_ske(torch.nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d_ske, self).__init__()

    def forward(self, logits, labels, skes, classweights, ske_weight):
        # targets: mask_gt
        # bkg. weight is 1, not listed out.
        gt_weight = 1.
        probs = torch.nn.functional.sigmoid(logits)
        probs_flat = probs.view(-1)
        mask_gt = labels.view(-1)
        ske = skes.view(-1)
        pos = mask_gt == 1
        neg = mask_gt == 0
        pos_ske = ske == 1
        neg_ske = ske == 0

        loss_bkg = torch.sum(classweights[0][1] * (torch.log(1 - probs_flat[neg] + 0.0001)))
        loss_mask = torch.sum(classweights[0][0] * torch.log(probs_flat[pos] * gt_weight + 0.0001))
        loss_ske = torch.sum(classweights[0][0] * (
                    mask_gt[pos_ske] * torch.log(probs_flat[pos_ske] * (ske_weight - gt_weight) + 0.0001))) + torch.sum(
            classweights[0][1] * ((1 - mask_gt[neg_ske]) * torch.log(1 - probs_flat[neg_ske] + 0.0001)))

        loss = -1. * (loss_mask + loss_bkg + loss_ske)

        return loss

# Measurement
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_prf(pred, mask, store='gpu'):
    '''
    per batch: tensor 'pred'/'mask': size: [40,1,242,242] --40 is batch size
    single 2d sample: size: [242,242]
    '''
    p = AverageMeter()
    r = AverageMeter()
    f = AverageMeter()

    if store == 'gpu':
        pred = pred.cpu().numpy()
        mask = mask.cpu().numpy()
    mutual = pred.squeeze() * mask
    precision = np.count_nonzero(mutual) / (np.count_nonzero(pred.squeeze()) + 1e-8)
    p.update(precision)
    recall = np.count_nonzero(mutual) / (np.count_nonzero(mask) + 1e-8)
    r.update(recall)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f.update(f1)

    return p.avg, r.avg, f.avg


# i/o functions
def psave(file_name, settings):
    with open(file_name, 'wb') as h:
        pickle.dump(settings, h, protocol=pickle.HIGHEST_PROTOCOL)
    print("Settings saved.")
    return

def pload(file_name):
    with open(file_name, 'rb') as h:
        settings = pickle.load(h)
    return settings

def fill_settings(settings):
    """ Update the rest of unspecified parameters in the configuration list """

    default_settings = {}
    default_settings['dataset'] = 'Not specified'
    default_settings['dataNormalization'] = 'localmax'
    default_settings['batchSize'] = 4
    default_settings['trafo'] = 'flip_augm'
    default_settings['img_size'] = (300, 300)
    default_settings['epochs'] = 1
    default_settings['learningRate'] = 1e-3
    default_settings['threshold'] = 0.5
    default_settings['validation_size'] = 0.2
    default_settings['savemodel'] = False
    default_settings['BCEWeights'] = [[2, 0.5]]
    default_settings['pretrained'] = None
    default_settings['architecture'] = 'UNet768'

    # Update hyperparameters
    if (settings):
        for key in default_settings.keys():
            if (key not in settings.keys()):
                settings[key] = default_settings[key]
    else:
        settings = default_settings

    return settings

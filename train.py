# custom libraries
from code_learning import model_builder
from code_learning import data_functions
from code_learning import helper_functions

# standard libraries
import scipy.io

# Example trainning:
# ch32 WeiSke5

# step 0： handling settings
settings = {}
settings_ = helper_functions.fill_settings(settings)
settings_['description'] = 'dsXY_242_rand_ch32'
settings_['architecture'] = 'UNet768_ch32'
settings_['FGRate'] = 0.91
settings_['batchSize'] = 17
settings_['epochs'] = 300
settings_['img_size'] = (242,242)
settings_['trainPids'] = scipy.io.loadmat('data/trainPids__' + settings_['description'] + '.mat')['trainPids'][0].tolist()
settings_['testPids'] = scipy.io.loadmat('data/testPids__' + settings_['description'] + '.mat')['testPids'][0].tolist()
helper_functions.psave("model/model_settings__" + settings_['description'] + ".pickledump",settings_)
settings = settings_

# step 1： start training
print('\n\n\n'+'training ch32 WeiSke5'+'\n\n\n')

# Create the model
net = model_builder.select_architecture(settings)
model = model_builder.Segmenter(net, settings)

# Dataset
trainLoader = data_functions.projection_loader_ske(settings, settings['trainPids'], 2, use_cuda=True,mode = 'train')

# Train
model_run = model.train_ske(trainLoader, settings)



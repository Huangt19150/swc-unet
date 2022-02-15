# custom libraries
from code_learning import model_builder
from code_learning import data_functions
from code_learning import helper_functions

# standard libraries
import torch

flag_test_2d = 1
test_2d_thresh = 0.6
savePath_2d = None #'/home/guolab/pythonScript/data/results_2d/results_2d_' + settings_['description'] + '/'
flag_test_3d = 1
test_3d_thresh = 0.5
savePath_3d = None #'/home/guolab/pythonScript/data/results_3d/results_3d_' + settings_['description'] + '/'

# load settings
settings = helper_functions.pload("model/model_settings.pickle")
# Create the model
net = model_builder.select_architecture(settings)
model = model_builder.Segmenter(net, settings)
model_file = '/home/guolab/pythonScript/model/final_300_step_200_250_51_defaultWei_134.pkl'
checkpoint = torch.load(model_file)
net.load_state_dict(checkpoint)

if flag_test_2d:
    testLoader = data_functions.projection_loader(settings, settings['testPids'], 2, use_cuda=True,mode = 'test')
    settings['threshold'] = test_2d_thresh
    savePath = savePath_2d
    model.test_2d(testLoader, settings, savePath)

if flag_test_3d:
    testLoader = data_functions.volumetric_loader(settings, settings['testPids'], 2, use_cuda=True)
    settings['threshold'] = test_3d_thresh
    savePath = savePath_3d
    labelPath_3d = '/home/guolab/pythonScript/data/labels_3d/'
    mode = 'average'
    probsvol_collect = model.test_3d(testLoader, labelPath_3d, settings, mode, savePath)


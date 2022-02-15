%% Multi-channal schedule paras
%%%%%%%%%% Enter here %%%%%%%%%%%%
scheduleName = 'dsXY_242_rand_ch32'; %'dsXY_242_rand';%
numCh = 32;
oriResize = [328, 328, 242];
subVolSize = 242;
numSamplePerStack = 4;
overlap = 0.1;

do_new_randomize = 1;
if_save_new = 1;
do_train = 1;
do_test = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run
% Initialize data path
rootDataPath = '/home/guolab/pythonScript/data/';

% Load data id
stackId = load([rootDataPath,'ori_stacks_3d/stackId.mat']);
stackId = stackId.stackId';
testStackId = [798,817,822,879,903,904,963,1011,1044,2083,2099,2116,2967,2974,2976]';
trainStackId = setdiff(stackId, testStackId);

% make/load schedule
if ~do_new_randomize
    % load pre-saved 'sample3dList'
    if exist([rootDataPath, 'sampleRefList__', scheduleName, '.mat'], 'file')
        load([rootDataPath, 'sampleRefList__', scheduleName, '.mat']);
    end
else
    outPaths = gen_paths(scheduleName);
    if do_train
        sample3dList_train = sub_vol_scheduler_random(trainStackId, oriResize, subVolSize, numSamplePerStack);
        sample3dList = sample3dList_train;
    end
    if do_test
        sample3dList_test = sub_vol_scheduler_regular(testStackId, oriResize, subVolSize, overlap);
        sample3dList(length(sample3dList_train)+1 : length(sample3dList_train)+length(sample3dList_test)) = sample3dList_test;
    end
end

% Run volume cutter
[trainPids, testPids] = volume_cutter_multiCh(sample3dList, stackId, oriResize, subVolSize, numCh, outPaths);

% Save pre-processing schedule
if if_save_new
    save([rootDataPath, 'trainPids__', scheduleName, '.mat'],'trainPids');
    save([rootDataPath, 'testPids__', scheduleName, '.mat'],'testPids');
    save([rootDataPath, 'sampleRefList__', scheduleName, '.mat'], 'sample3dList');
end


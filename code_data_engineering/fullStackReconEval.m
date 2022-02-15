%% Schedule paras for Evaluation 
%%%%%%%%%% Enter here %%%%%%%%%%%%
scheduleName = 'dsXY_242_rand_ch32';
oriResize = [328, 328, 242];
subVolSize = 242;
numSamplePerStack = 4;
overlap = 0.1;

% Option: 1,2,3: will WRITE reconstructed full stack to disk
% with different name tag for convenience (see function)
saveMode = 0; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Run
rootDataPath = '/home/guolab/pythonScript/data/';
stackId = load([rootDataPath,'ori_stacks_3d/stackId.mat']);
stackId = stackId.stackId';
testStackId = [798,817,822,879,903,904,963,1011,1044,2083,2099,2116,2967,2974,2976]';
trainStackId = setdiff(stackId, testStackId);

outPaths = gen_paths(scheduleName); 

[precision, recall, f1] = compute_full_stack_prf(scheduleName, oriResize, subVolSize, outPaths, saveMode);
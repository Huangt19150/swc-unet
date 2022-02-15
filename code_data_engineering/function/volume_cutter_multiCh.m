function [trainPids, testPids] = volume_cutter_multiCh(sample3dList, stackId, oriResize, subVolSize, numCh, outPaths)

%
rootDataPath = '/home/guolab/pythonScript/data/';
inputSamplePrefix = 'data_patch_';
labelSamplePrefix = 'label_patch_';
skeSamplePrefix = 'ske_patch_';

inputImagePath = [rootDataPath, 'ori_stacks_3d/']; % fixed
%imgList = dir([inputImagePath,'stack_*.tif']); % ...write name with 'StackId' instead
inputLabelPath = [rootDataPath, 'ori_labels_3d/']; % fixed.............................................adjust if needed
%inputLabelPath = [rootDataPath, 'ori_skeletons_3d/']; 
inputSkePath = [rootDataPath, 'ori_skeletons_3d/']; % fixed
%skeList = dir([inputSkePath,'mask_*_Dil2.tif']);


%%% stackId
stackId_str = num2str(stackId, '%05d');


%%%
tic
parpool(15);
parfor i = 1:length(stackId_str)
    stack0 = loadTifFast([inputImagePath, 'stack_',stackId_str(i,:),'.tif']);% imgList(i).name
    label0 = loadTifFast([inputLabelPath, 'mask_',stackId_str(i,:),'_DilSphere4.tif']);% labelList(i).name.......adjust accordingly
    ske0 = loadTifFast([inputSkePath, 'mask_',stackId_str(i,:),'_DilSphere2.tif']);% labelList(i).name
    
    BG = median(stack0,3);
    stack_subBG = stack0 - repmat(BG,[1 1 242]);% background subtraction
    stack_subBG_med = medfilt3(stack_subBG);% reduce noise by blurring
    
%     figure,imshow(max(stack_subBG_med,[],3),[]);
%     figure,histogram(stack_subBG_med);
    stack_resize = imresize3(stack_subBG_med, oriResize);
    
%     figure,histogram(stack_resize);
%     figure,imshow(max(stack_resize,[],3),[]);

    highPass = stack_resize > 200;
    lowPass = stack_resize <= 200;
    stack_lowPass = uint16(highPass).*200 + uint16(lowPass).*stack_resize;
%     figure,imshow(max(stack_lowPass,[],3),[]);
    %
    MEAN = double(mean(stack_lowPass,'all'));
    STD = std(double(stack_lowPass),0,'all');
    stack_norm = (double(stack_lowPass) - MEAN)./ STD;
%     figure,imshow(max(stack_norm,[],3),[]);
    MAX = double(max(stack_norm, [],'all'));
    MIN = double(min(stack_norm, [],'all'));
    stack = ((stack_norm - MAX)./(MAX - MIN))+1;
%     figure,imshow(max(stack,[],3),[]);
%     figure,histogram(stack_norm);
    %
    label = imresize3(label0, oriResize, 'nearest');
    ske = imresize3(ske0, oriResize, 'nearest');
    %
    for sample_i = 1:length(sample3dList)
        if strcmp(sample3dList(sample_i).stackId, stackId_str(i,:))
            ori = sample3dList(sample_i).refPoint;
            crop_stack = stack(ori(1): ori(1) + subVolSize - 1,...
                               ori(2): ori(2) + subVolSize - 1,...
                               ori(3): ori(3) + subVolSize - 1);
            crop_label = label(ori(1): ori(1) + subVolSize - 1,...
                               ori(2): ori(2) + subVolSize - 1,...
                               ori(3): ori(3) + subVolSize - 1);
            crop_ske = ske(ori(1): ori(1) + subVolSize - 1,...
                           ori(2): ori(2) + subVolSize - 1,...
                           ori(3): ori(3) + subVolSize - 1);                           
            % save input 
            writeNPY(multi_channel_max(crop_stack,[subVolSize,subVolSize,numCh]),...
                     [outPaths.inputPath, inputSamplePrefix, num2str(sample_i-1),'_Z.npy']);
            writeNPY(multi_channel_max(crop_stack,[subVolSize,numCh,subVolSize]),...
                     [outPaths.inputPath, inputSamplePrefix, num2str(sample_i-1),'_X.npy']);
            writeNPY(multi_channel_max(crop_stack,[numCh,subVolSize,subVolSize]),...
                     [outPaths.inputPath, inputSamplePrefix, num2str(sample_i-1),'_Y.npy']);
                           
            % save labels ........imresize3() won't keep ones in labels!!!
            writeNPY(multi_channel_max(crop_label,[subVolSize,subVolSize,numCh]),...
                     [outPaths.labelPath, labelSamplePrefix, num2str(sample_i-1),'_Z.npy']);
            writeNPY(multi_channel_max(crop_label,[subVolSize,numCh,subVolSize]),...
                     [outPaths.labelPath, labelSamplePrefix, num2str(sample_i-1),'_X.npy']);
            writeNPY(multi_channel_max(crop_label,[numCh,subVolSize,subVolSize]),...
                     [outPaths.labelPath, labelSamplePrefix, num2str(sample_i-1),'_Y.npy']);
            
            % save labels_3d
            writeNPY(crop_label,...
                     [outPaths.label3dPath, labelSamplePrefix, num2str(sample_i-1),'.npy']);
            writeTifFast([outPaths.label3dPath, labelSamplePrefix, num2str(sample_i-1),'.tif'],...
                         crop_label.*255, 8); %.*255, for vis.
                     
            % save skeletons
            writeNPY(multi_channel_max(crop_ske,[subVolSize,subVolSize,numCh]),...
                     [outPaths.skeletonPath, skeSamplePrefix, num2str(sample_i-1),'_Z.npy']);
            writeNPY(multi_channel_max(crop_ske,[subVolSize,numCh,subVolSize]),...
                     [outPaths.skeletonPath, skeSamplePrefix, num2str(sample_i-1),'_X.npy']);
            writeNPY(multi_channel_max(crop_ske,[numCh,subVolSize,subVolSize]),...
                     [outPaths.skeletonPath, skeSamplePrefix, num2str(sample_i-1),'_Y.npy']);
                     
                     
        end
        
    end
    
end
delete(gcp('nocreate'));

fprintf(['Sampling took: ', num2str(toc/60), ' minutes.\n']);

%%%
trainPids = [];
testPids = [];
for sample_i = 1:length(sample3dList)
    if strcmp(sample3dList(sample_i).usageTag, 'train')
        trainPids = [trainPids, (sample_i-1)];
    elseif strcmp(sample3dList(sample_i).usageTag, 'test')
        testPids = [testPids, (sample_i-1)];
    end
end


end
function [precision, recall, f1] = compute_full_stack_prf(scheduleName, oriResize, subVolSize, outPaths, saveMode)

%
rootDataPath = '/home/guolab/pythonScript/data/';
load([rootDataPath, 'sampleRefList__', scheduleName, '.mat'], 'sample3dList');
if saveMode == 1
    savePath = outPaths.resultrRecovPath;
    tag = '_defaultWei';
elseif saveMode == 2
    savePath = outPaths.resultrRecovPath;
    tag = '_WeiSke';
elseif saveMode == 3
    savePath = outPaths.resultrRecovPath;
    tag = '_patch';
end
subVolList = dir([outPaths.result3dPath,'*',tag,'*','.npy']);

%
stackId = [];
threshold = [0.5,0.6,0.7,0.8,0.9];
sum_p = zeros(1,5,'double');
sum_r = zeros(1,5,'double');
sum_f = zeros(1,5,'double');
count = 0;

for i = 1:length(subVolList)
    
    % get every pid   
    splitName = split(subVolList(i).name(1:15),'_');
    pid = str2double(splitName{3,1});
    
    % load volume
    subVol = double(readNPY([outPaths.result3dPath, subVolList(i).name]));
    %
    if ~strcmp(sample3dList(pid+1).stackId, stackId)
        if i ~= 1
            stack_oriSize = imresize3(stack,[1200,1200,242]);
            if saveMode
                writeTifFast([savePath, 'prob_', stackId, tag, '.tif'], stack_oriSize, 8);
            end
            for thI = 1:5
                pred = stack_oriSize > threshold(thI);
                [p, r, f] = compute_prf(pred, label);
                sum_p(thI) = sum_p(thI)+p;
                sum_r(thI) = sum_r(thI)+r;
                sum_f(thI) = sum_f(thI)+f;
            end
        end
        count = count +1;
        stackId = sample3dList(pid+1).stackId;
        stack = zeros(oriResize(1),oriResize(2),oriResize(3), 'double');
        label = loadTifFast([rootDataPath, 'ori_labels_3d/mask_', stackId, '_DilSphere4.tif']);
    else
        x0 = sample3dList(pid+1).refPoint(1);
        y0 = sample3dList(pid+1).refPoint(2);
        z0 = sample3dList(pid+1).refPoint(3);
        stack(x0:x0+subVolSize-1, y0:y0+subVolSize-1, z0:z0+subVolSize-1)=...
            max(stack(x0:x0+subVolSize-1, y0:y0+subVolSize-1, z0:z0+subVolSize-1),...
                subVol);
        if i == length(subVolList)
            stack_oriSize = imresize3(stack,[1200,1200,242]);
            if saveMode
                writeTifFast([savePath, 'prob_', stackId, tag, '.tif'], stack_oriSize, 8);
            end

            for thI = 1:5
                pred = stack_oriSize > threshold(thI);
                [p, r, f] = compute_prf(pred, label);
                sum_p(thI) = sum_p(thI)+p;
                sum_r(thI) = sum_r(thI)+r;
                sum_f(thI) = sum_f(thI)+f;
            end
        end
    end
end

for printI = 1:length(threshold)
    precision = sum_p(1,printI)/count;
    recall = sum_r(1,printI)/count;
    f1 = sum_f(1,printI)/count;

    fprintf(['threshold: ',num2str(threshold(printI)),'\n']);
    fprintf(['precision: ', num2str(precision), ', recall: ', num2str(recall), ', f1 score: ', num2str(f1), '\n']);
end

end






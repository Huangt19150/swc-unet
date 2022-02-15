function sample3dList = sub_vol_scheduler_random(trainStackId, oriResize, subVolSize, numSamplePerStack)

% HTY, 20/01/15
% refPoint to a sub-volume is selected as the Upper-left corner 
% outPaths: see out-put of the function 'gen_paths'


% refPoint
refPointRange = struct();
refPointRange.d1 = oriResize(1) - subVolSize + 1;
refPointRange.d2 = oriResize(2) - subVolSize + 1;
refPointRange.d3 = oriResize(3) - subVolSize + 1;

%
trainStackId_str = num2str(trainStackId, '%05d');


%%% Randomly generate a sampling reference list.
sample3dList = struct('stackId', [], 'refPoint', [], 'usageTag', []);

count = 0;
for s = 1:length(trainStackId)
    
    % generate random refPoints
    randD1 = randperm(refPointRange.d1, numSamplePerStack);
    randD2 = randperm(refPointRange.d2, numSamplePerStack);
    if refPointRange.d3 >= numSamplePerStack
        randD3 = randperm(refPointRange.d3, numSamplePerStack);
    else
        randD3 = ones(numSamplePerStack);
    end
    
    % pack into sample3dList
    for sample_i = 1: numSamplePerStack
        count = count +1;
        sample3dList(count).refPoint = ...
            [randD1(sample_i), randD2(sample_i), randD3(sample_i)];
        sample3dList(count).stackId = trainStackId_str(s,:);
    end
            
end


% for i = 1:numSample3d
%     stack_idx = floor((i-1)/numSamplePerStack) + 1; % index of stack in 'trainStackId_str'
% %     strings = strsplit(imgList(stack_idx).name, {'_','.'});
%     sample3dList(i).trainStackId = trainStackId_str(stack_idx,:);
%     
%     if mod(i-1, numSamplePerStack)==0
%         randX = randperm(refPointRange.x, numSamplePerStack);
%         randY = randperm(refPointRange.y, numSamplePerStack);
%         if refPointRange.z >= numSamplePerStack
%             randZ = randperm(refPointRange.z, numSamplePerStack);
%         else
%             randZ = ones(numSamplePerStack);
%         end
%     end
%     sample3dList(i).refPoint = [randX(mod(i-1, numSamplePerStack)+1),...
%                                 randY(mod(i-1, numSamplePerStack)+1),...
%                                 randZ(mod(i-1, numSamplePerStack)+1)];
% end

[sample3dList.usageTag] = deal('train');


end
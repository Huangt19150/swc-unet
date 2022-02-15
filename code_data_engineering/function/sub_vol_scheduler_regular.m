function sample3dList = sub_vol_scheduler_regular(testStackId, oriResize, subVolSize, overlap)

% HTY, 20/01/15
% refPoint to a sub-volume is selected as the Upper-left corner 
% outPaths: see out-put of the function 'gen_paths'

% 
testStackId_str = num2str(testStackId, '%05d');


%%% Regularly generate a sampling reference list.
sample3dList = struct('stackId', [], 'refPoint', [], 'usageTag', []);

ref_d1 = [1: round(subVolSize*(1-overlap)-1) :oriResize(1)];
ref_d2 = [1: round(subVolSize*(1-overlap)-1) :oriResize(2)];
ref_d3 = [1: round(subVolSize*(1-overlap)-1) :oriResize(3)];

count = 0;
for s = 1:length(testStackId) 
    for d1 = 1:length(ref_d1)
        if d1 == length(ref_d1)
            d1_val = oriResize(1)-subVolSize+1;
        else
            d1_val = ref_d1(d1);
        end
        
        for d2 = 1:length(ref_d2)
            if d2 == length(ref_d2)
                d2_val = oriResize(2)-subVolSize+1;
            else
                d2_val = ref_d2(d2);
            end
            
            for d3 = 1:length(ref_d3)
                count = count+1;                
                if d3 == length(ref_d3)
                    d3_val = oriResize(3)-subVolSize+1;
                else
                    d3_val = ref_d3(d3);                   
                end
                sample3dList(count).refPoint = [d1_val, d2_val, d3_val];
                sample3dList(count).stackId = testStackId_str(s,:);
            end
        end
    end
    
end

[sample3dList.usageTag] = deal('test');


end
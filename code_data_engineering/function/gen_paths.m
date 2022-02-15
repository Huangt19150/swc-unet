function outPaths = gen_paths(newSamplingScheduleAsFoldername)

% HTY, 20/01/15
% generate default paths arrangements for a new sampling schedule.
% newSamplingScheduleAsFoldername: make a SHORT string to represent the
% sampling schedule.

rootDataPath = '/home/guolab/pythonScript/data/';

outPaths = struct();
outPaths.inputPath = [rootDataPath,'input/input_',newSamplingScheduleAsFoldername,'/'];
mkdir(outPaths.inputPath);
outPaths.labelPath = [rootDataPath,'labels/labels_',newSamplingScheduleAsFoldername,'/'];
mkdir(outPaths.labelPath);
outPaths.label3dPath = [rootDataPath,'labels_3d/labels_3d_',newSamplingScheduleAsFoldername,'/'];
mkdir(outPaths.label3dPath);
outPaths.skeletonPath = [rootDataPath,'skeletons/skeletons_',newSamplingScheduleAsFoldername,'/'];
mkdir(outPaths.skeletonPath);
outPaths.skeleton3dPath = [rootDataPath,'skeletons_3d/skeletons_3d_',newSamplingScheduleAsFoldername,'/'];
mkdir(outPaths.skeleton3dPath);
outPaths.result2dPath = [rootDataPath,'results_2d/results_2d_',newSamplingScheduleAsFoldername,'/'];
mkdir(outPaths.result2dPath);
outPaths.result3dPath = [rootDataPath,'results_3d/results_3d_',newSamplingScheduleAsFoldername,'/'];
mkdir(outPaths.result3dPath);
outPaths.resultrRecovPath = [rootDataPath,'results_recov_stack/results_recov_stack_',newSamplingScheduleAsFoldername,'/'];
mkdir(outPaths.resultrRecovPath);

outPaths;
fprintf('mkdir done.\n');

end
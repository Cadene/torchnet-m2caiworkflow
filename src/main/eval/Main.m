close all; clear all;

phaseGroundTruths = {'workflow_video_02.txt', ...
    'workflow_video_10.txt', ...
    'workflow_video_27.txt', ...
    'workflow_video_09.txt', ...
    'workflow_video_13.txt'};

phases = {'TrocarPlacement', 'Preparation',  'CalotTriangleDissection', ...
    'ClippingCutting', 'GallbladderDissection',  'GallbladderPackaging', 'CleaningCoagulation', ...
    'GallbladderRetraction'};

fps = 25;

for i = 1:length(phaseGroundTruths)

    phaseGroundTruth = ['raw/annotations/' phaseGroundTruths{i}];
    predFile = ['experiments/hmmval/dirname/hmm/' phaseGroundTruths{i}(1:end-4) '_pred.txt'];

    disp(phaseGroundTruth)
    disp(predFile)

    [gt] = ReadPhaseLabel(phaseGroundTruth);
    [pred] = ReadPhaseLabel(predFile);

    if(size(gt{1}, 1) ~= size(pred{1},1) || size(gt{2}, 1) ~= size(pred{2},1))
        error(['ERROR:' ground_truth_file '\nGround truth and prediction have different sizes']);
    end

    if(~isempty(find(gt{1} ~= pred{1})))
        error(['ERROR: ' ground_truth_file '\nThe frame index in ground truth and prediction is not equal']);
    end

    % reassigning the phase labels to numbers
    gtLabelID = [];
    predLabelID = [];
    for j = 1:length(phases)
        gtLabelID(find(strcmp(phases{j}, gt{2}))) = j;
        predLabelID(find(strcmp(phases{j}, pred{2}))) = j;
    end

    % compute jaccard index and the accuracy
    [jaccard(:,i), acc(i)] = Evaluate(gtLabelID, predLabelID, fps);

end

meanPerPhase = nanmean(jaccard, 2);
meanJacc = mean(meanPerPhase);
stdJacc = std(meanPerPhase);
meanAcc = mean(acc);
stdAcc = std(acc);

disp('============================================');
disp('             Results (Jaccard)              ');
disp('============================================');
for iPhase = 1:length(phases)
    disp([sprintf('%25s',phases{iPhase}) ': ' sprintf('%5.2f', meanPerPhase(iPhase))]);
    disp('---------------------------------------------');
end
disp('============================================');
disp(['Mean jaccard: ' sprintf('%5.2f', meanJacc) ' +- ' sprintf('%5.2f', stdJacc)]);
disp(['Mean accuracy: ' sprintf('%5.2f', meanAcc) ' +- ' sprintf('%5.2f', stdAcc)]);

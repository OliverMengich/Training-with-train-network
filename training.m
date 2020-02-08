% Get training Images
wildcat_ds = imageDatastore('wild_cat','IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,testImgs] = splitEachLabel(wildcat_ds,0.6);
numClasses = numel(categories(wildcat_ds.Labels))
% Create A network by modifying Alexnet
net = alexnet;
layers = net.Layers
imageSize = net.Layers(1).InputSize
AugmentedTrainingSet = augmentedImageDatastore(imageSize,trainImgs,...
    'ColorPreprocessing','gray2rgb');
AugmentedTestSet = augmentedImageDatastore(imageSize,testImgs,...
    'ColorPreprocessing','gray2rgb');
layers(end-2) = fullyConnectedLayer(numClasses);
layers(end) = classificationLayer;
% Set trainig Algorithm Options
options = trainingOptions('sgdm','InitialLearnRate',0.001);
output = net.Layers(end).OutputSize;
output = numClasses
% The Training
[flowerNet,info] = trainNetwork(trainImgs,layers,options);
% Use trained Network To classify test Images
testPreds = classify(flowerNet,TestImages);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%    File:   JaredNN.m
%    Author: Jared Johansen
% 
%    POSSIBLE FUTURE IMPROVEMENTS
%    add a bias at each node
%    let the user specify the NN parameters (modularize code better)
%    let the user specify the activation function
%    support arbitrary XOR size
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FRONT MATTER
% clear variables/terminal
clc;
clear all;

% user-specified values
dataSize = 30000;
trainRatio = 0.9;
testRatio = 0.1;
learningRate = 0.2;
learningRateDecay = 1.0;
learningRateDecayFrequency = 2000; % reduce learning rate each X iterations 
numHiddenNodes = 10;
printErrFreq = 1000; % print err every X iterations

%% CREATE DATA
% create input data
xt = randi([0 1], 2, dataSize);

% create desired output
dt = zeros(1,dataSize);
for i=1:dataSize
	dt(i) = mod(sum(xt(:,i)),2);
end

% train/test division
trainingDataNum = trainRatio * dataSize;
trainData = xt(:, 1:trainingDataNum);
trainLabels = dt(:, 1:trainingDataNum);
testData = xt(:, trainingDataNum+1:end);
testLabels = dt(:, trainingDataNum+1:end);

%% NN SETUP
% NN parameters
numInputs = 2;
numOutputs = 1;
numHiddenLayers = 1; % not used

% input weights
iWeights = rand(numHiddenNodes, numInputs);

% output weights
oWeights = rand(numHiddenNodes, numOutputs);

% keep track of errors
errVals = zeros(1, trainingDataNum);


%% TRAIN
fprintf('Training!\n')

% loop through train data
for i=1:size(trainData,2)
   
    %%%% FORWARD PROPAGATION -- Hidden Layer %%%%
    % calculate summation of weights: E(w(x) * i(x))
    sumsHidden = iWeights*trainData(:,i);
    % calculate the sigmoid output: f(x) = 1 / (1+e^-x)
    za = 1./(1+exp(-sumsHidden));
    
    %%%% FORWARD PROPAGATION -- Output Layer %%%%
    % calculate summation of weights: E(w(x) * i(x))
    sumsOut = oWeights' * za;
    % calculate the sigmoid output: f(x) = 1 / (1+e^-x)
    y = 1/(1+exp(-sumsOut));
    	
	% error calculation
    errorOut = trainLabels(i) - y;
    errVals(i) = abs(errorOut);
    
    % print data every printErrFreq iterations
    if (mod(i, printErrFreq) == 0)
     	fprintf('iter: %d\terr: %f\n', i, abs(errorOut));
%         fprintf('input weights:\n');
%         display(iWeights);
%         fprintf('output weights:\n');
%         display(oWeights);
    end
    
    %%%% BACKPROPAGATION -- Calc output delta %%%%
    % f'(x) = (e^-x)/(1+e^-x)^2 = f(x) * (1-f(x))
    fPrimeOut = (y * (1-y)); 
    % delta = error * f'(x)    
    deltaOut = errorOut * fPrimeOut; 

    %%%% BACKPROPAGATION -- Calc hidden layer delta %%%%
    % f'(x) = (e^-x)/(1+e^-x)^2 = f(x) * (1-f(x))
    fPrimeHidden = za .* (ones(numHiddenNodes,1) - za); 
    % error = weight * prevLayerDelta
    errorHidden = oWeights .* deltaOut;
    % delta = error * f'(x)
    deltaHidden = errorHidden .* fPrimeHidden;
    
    %%%% BACKPROPAGATION -- Update input weights %%%%
    iWeights = iWeights + (learningRate * deltaHidden * (trainData(:,i)'));
	
    %%%% BACKPROPAGATION -- Update output weights %%%%
	oWeights =	oWeights + learningRate * deltaOut * za;   
    
    % update learning rate
    if (mod(i,learningRateDecayFrequency) == 0)
        learningRate = learningRate * learningRateDecay;
    end
end

%% TEST
fprintf('Testing!\n')
correct = 0;
total = dataSize - trainingDataNum;

% loop through test data
for i=1:size(testData,2)
    %%%% FORWARD PROPAGATION -- Hidden Layer %%%%
    % calculate summation of weights: E(w(x) * i(x))
    x = iWeights*testData(:,i);
    % calculate the sigmoid output: f(x) = 1 / (1+e^-x)
    za = 1./(1+exp(-x)); 

    %%%% FORWARD PROPAGATION -- Output Layer %%%%
    % calculate summation of weights: E(w(x) * i(x))
    x = oWeights' * za;
    % f(x) = 1 / (1+e^-x)
   	fx = 1/(1+exp(-x));    
    % step function (to output class 0 or 1)
    y = round(fx); 
    
	% keep track of the number of correct predictions
	if (y == testLabels(i))
		correct = correct + 1;
	end	
end

% display results
percent = 1.0 * correct / total;
fprintf('Correct: %d/%d = %f\n', correct, total, percent)

%% DISPLAY PLOT
plot(errVals);
title('Output Error: XOR NN');
xlabel('Training Data Iteration #');
ylabel('Output Error');


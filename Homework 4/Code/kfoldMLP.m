function dummyOut = kfoldMLP(N)
% Incomplete - do not use!

% Maximum likelihood training of a 2-layer MLP
% assuming additive (white) Gaussian noise
close all, 
dummyOut = 0;
% Input N specifies number of training samples

% Generate data using a Gaussian Mixture Distribution
%mu = [1 2;-7 0];
%Sigma = cat(3,[4 0.9; 0.9 0.5],[5 0; 0 0.25]);
%mixp = ones(1,2)/2;
%gm = gmdistribution(mu,Sigma,mixp);
%data = [random(gm,N)';sqrt(0.2)*randn(1,N)];
%figure(1), clf, plot3(data(1,:),data(2,:),data(3,:),'.'); axis equal,
%X = [data(1,:);data(3,:)]; Y = data(2,:);

% Determine/specify sizes of parameter matrices/vectors
nX = 2;%size(X,1); 
nPerceptrons = 5; 
nY = 1;%size(Y,1);
sizeParams = [nX;nPerceptrons;nY]
K = 5; % for K-fold c.v.

% Generate training data
x = 10*randn(nX,N);
paramsTrue.A = 0.3*rand(nPerceptrons,nX)
paramsTrue.b = 0.3*rand(nPerceptrons,1);
paramsTrue.C = 0.3*rand(nY,nPerceptrons);
paramsTrue.d = 0.3*rand(nY,1);
y = mlpModel(X,paramsTrue)+1e-14*randn(nY,N);
vecParamsTrue = [paramsTrue.A(:);paramsTrue.b;paramsTrue.C(:);paramsTrue.d];
%figure(1), clf, plot3(x(1,:),x(2,:),y(1,:),'.g');

% Divide the data set into K approximately-equal-sized partitions
dummy = ceil(linspace(0,N,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end
% Allocate space
MSEtrain = zeros(K,N); MSEvalidate = zeros(K,N); 
AverageMSEtrain = zeros(1,N); AverageMSEvalidate = zeros(1,N);
% Try several numbers-of-perceptrons
for M = 1:5
    % K-fold cross validation
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = x(indValidate); % Using folk k as validation set
        yValidate = y(indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k+1,1):N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k-1,2)];
        else
            indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
        end
        xTrain = x(indTrain); % using all other folds as training set
        yTrain = y(indTrain);
        Ntrain = length(indTrain); Nvalidate = length(indValidate);
        % Train model parameters
        % Initialize model parameters
        params.A = 0*paramsTrue.A+1e-1*randn(M,nX);
        params.b = 0*paramsTrue.b+1e-1*randn(M,1);
        params.C = 0*paramsTrue.C+1e-1*randn(nY,M);
        params.d = mean(Y,2);%zeros(nY,1); % initialize to mean of y
        %params = paramsTrue;
        vecParamsInit = [params.A(:);params.b;params.C(:);params.d];
        %vecParamsInit = vecParamsTrue; % Override init weights with true weights
        % Optimize model
        options = optimset('MaxFunEvals',1e4*length(vecParamsInit)); % Matlab default is 200*length(vecParamsInit)
        vecParams = fminsearch(@(vecParams)(objectiveFunction(xTrain,yTrain,sizeParams,vecParams)),vecParamsInit,options);
        % Visualize model output for training data
        params.A = reshape(vecParams(1:nX*M),M,nX);
        params.b = vecParams(nX*M+1:(nX+1)*M);
        params.C = reshape(vecParams((nX+1)*M+1:(nX+1+nY)*M),nY,M);
        params.d = vecParams((nX+1+nY)*M+1:(nX+1+nY)*M+nY);
        hValidate = mlpModel(xValidate,params);
        [~,dValidate] = max(hValidate,[],1); % MAP decision rule
        [~,lValidate] = max(yValidate,[],1); % MAP decision rule
        MSEvalidate(k,M) = sum(sum((yValidate-hValidate).*(yValidate-hValidate),1),2)/N;
        %pErrorVal(k,M) = 1-length(find(dValidate==lValidate))/Nvalidate;
    end
    %    AverageMSEtrain(1,M) = mean(MSEtrain(:,M)); % average training MSE over folds
    AverageMSEvalidate(1,M) = mean(MSEvalidate(:,M)); % average validation MSE over folds
    %pErrorVal(1,M) = mean(pError,1);
end
%[~,bestM] = min(AverageMSEvalidate);
[~,bestM] = min(pErrorVal);
% Train model parameters
% Initialize model parameters
params.A = 0*paramsTrue.A+1e-1*randn(bestM,nX);
params.b = 0*paramsTrue.b+1e-1*randn(bestM,1);
params.C = 0*paramsTrue.C+1e-1*randn(nY,bestM);
params.d = mean(Y,2);%zeros(nY,1); % initialize to mean of y
%params = paramsTrue;
vecParamsInit = [params.A(:);params.b;params.C(:);params.d];
%vecParamsInit = vecParamsTrue; % Override init weights with true weights
% Optimize model
options = optimset('MaxFunEvals',1e4*length(vecParamsInit)); % Matlab default is 200*length(vecParamsInit)
vecParams = fminsearch(@(vecParams)(objectiveFunction(x,y,sizeParams,vecParams)),vecParamsInit,options);

% figure(2), clf, plot(Y,H,'.'); axis equal,
% xlabel('Desired Output'); ylabel('Model Output');
% title('Model Output Visualization For Training Data')
% vecParamsFinal = [params.A(:);params.b;params.C(:);params.d];
% figure(1), hold on, plot3(X(1,:),X(2,:),H(1,:),'.r');
% xlabel('X_1'), ylabel('X_2'), zlabel('Y and H'),
% [vecParamsTrue,vecParamsInit,vecParamsFinal]
% keyboard,

%%%
function objFncValue = objectiveFunction(X,Y,sizeParams,vecParams)
N = size(X,2); % number of samples
nX = sizeParams(1);
nPerceptrons = sizeParams(2);
nY = sizeParams(3);
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
H = mlpModel(X,params);
% Change the objective function appropriately
objFncValue = sum(sum((Y-H).*(Y-H),1),2)/N; % MSE for regression under AWGN model
%objFncValue = sum(-sum(Y.*log(H),1),2)/N; % CrossEntropy for ClassPosterior approximation

%%%
function H = mlpModel(X,params)
N = size(X,2);                          % number of samples
nY = length(params.d);                  % number of outputs
U = params.A*X + repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
Z = activationFunction(U);              % z \in R^nP, using nP instead of nPerceptons
V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
H = V; % linear output layer activations
%H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer
% Activate the softmax function to make this MLP a model for class posteriors
%
function out = activationFunction(in)
% Pick a shared nonlinearity for all perceptrons: sigmoid or ramp style...
% You can mix and match nonlinearities in the model.
% However, typically this is not done; identical nonlinearity functions
% are better suited for parallelization of the implementation.
%out = 1./(1+exp(-in)); % Logistic function - sigmoid style nonlinearity
out = in./sqrt(1+in.^2); % ISRU - ramp style nonlinearity




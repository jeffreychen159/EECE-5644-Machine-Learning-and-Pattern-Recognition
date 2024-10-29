function dummyOut = mleMLPwAWGN(N)

% Maximum likelihood training of a 2-layer MLP
% assuming additive (white) Gaussian noise
close all, 
dummyOut = 0;

T_train_20 = table2array(readtable('d20_train.csv'))';
T_10k_validate = table2array(readtable('d10k_validate.csv'))';

% Determine/specify sizes of parameter matrices/vectors
nX = 2;%size(X,1); 
nPerceptrons = 5; 
nY = 1;%size(Y,1);
sizeParams = [nX;nPerceptrons;nY];

X = T_train_20(1:2,:);
paramsTrue.A = 0.3*rand(nPerceptrons,nX);
paramsTrue.b = 0.3*rand(nPerceptrons,1);
paramsTrue.C = 0.3*rand(nY,nPerceptrons);
paramsTrue.d = 0.3*rand(nY,1);
Y = T_train_20(3,:);
vecParamsTrue = [paramsTrue.A(:);paramsTrue.b;paramsTrue.C(:);paramsTrue.d];

% Initialize model parameters
params.A = 0*paramsTrue.A+1e-1*randn(nPerceptrons,nX);
params.b = 0*paramsTrue.b+1e-1*randn(nPerceptrons,1);
params.C = 0*paramsTrue.C+1e-1*randn(nY,nPerceptrons);
params.d = mean(Y,2);%zeros(nY,1); % initialize to mean of y
%params = paramsTrue;
vecParamsInit = [params.A(:);params.b;params.C(:);params.d];
%vecParamsInit = vecParamsTrue; % Override init weights with true weights

% Optimize model
options = optimset('MaxFunEvals',1e4*length(vecParamsInit)); % Matlab default is 200*length(vecParamsInit)
% MaxFunEvals is the maximum number of function evaluations you allow
% fminsearch to conduct during parameter optimization. Setting it to a
% larger value allows the optimization algorithm to spend more time fine
% tuning the parameters towards a local optimum. You should see the effect
% of smaller MaxFunEvals as worse performance of final model. The drawback
% of using a larger value is the possibility of training taking longer.
vecParams = fminsearch(@(vecParams)(objectiveFunction(X,Y,sizeParams,vecParams)),vecParamsInit,options);

% Visualize model output for training data
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);

H = [];

% Generates 10000 samples to test against validate
for i = 1:500
    H = [H mlpModel(X,params)];
end

rounded = zeros(1,10000);

% Takes the values and rounded them to the nearest class posterior
for i = 1:10000
    check_0 = abs(H(i) - 0);
    check_1 = abs(H(i) - 1);
    if check_0 < check_1
        rounded(i) = 0;
    else
        rounded(i) = 1;
    end
end

% Writes both the actual and rounded class posteriors to a csv
writematrix(H, 'actual_t20_log.csv');
writematrix(rounded, 'rounded_t20_log.csv');

validate_labels = T_10k_validate(3,:);

actual_matches = length(find(H == validate_labels));
rounded_matches = length(find(rounded == validate_labels));

accuracy_rate_actual = actual_matches/10000;
accuracy_rate_rounded = rounded_matches/10000;

accuracy_rate_actual, 
accuracy_rate_rounded, 
error_d20 = 1 - accuracy_rate_rounded;
error_d20,


%figure(2), clf, plot(Y,H,'.'); axis equal,
%xlabel('Desired Output'); ylabel('Model Output');
%title('Model Output Visualization For Training Data')
%vecParamsFinal = [params.A(:);params.b;params.C(:);params.d];
%figure(1), hold on, plot3(X(1,:),X(2,:),H(1,:),'.r')n 
%xlabel('X_1'), ylabel('X_2'), zlabel('Y and H'),
%[vecParamsTrue,vecParamsInit,vecParamsFinal]
keyboard,

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
% objFncValue = sum(-sum(Y.*log(H),1),2)/N; % CrossEntropy for ClassPosterior approximation

%%%
function H = mlpModel(X,params)
N = size(X,2);                        % number of samples
nY = length(params.d);                  % number of outputs
U = params.A*X + repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
Z = activationFunction(U);              % z \in R^nP, using nP instead of nPerceptons
V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
H = V; % linear output layer activations
% H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer
% Activate the softmax function to make this MLP a model for class posteriors
%
function out = activationFunction(in)
% Pick a shared nonlinearity for all perceptrons: sigmoid or ramp style...
% You can mix and match nonlinearities in the model.
% However, typically this is not done; identical nonlinearity functions
% are better suited for parallelization of the implementation.
out = 1./(1+exp(-in)); % Logistic function - sigmoid style nonlinearity
% out = in./sqrt(1+in.^2); % ISRU - sigmoid style nonlinearity




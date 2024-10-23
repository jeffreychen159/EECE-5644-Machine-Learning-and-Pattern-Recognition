% Expected risk minimization with C classes
clear all, close all,

C = 3;
N = 10000; % Number of samples
n = 3; % Data dimensionality (must be 2 for plots to work)

% Specify mean vectors (manually chosen for desired separations)
gmmParameters.priors = [0.3, 0.3, 0.4];
gmmParameters.meanVectors = [0,0,0;3,3,3;-3,-3,-3;6,-6,0]';

% Define covariance matrices for each Gaussian component
cov_base = 2 * eye(n);
gmmParameters.covMatrices(:,:,1) = [2.5 ,-0.2, 0.1;
                                    -0.2,1.8,-0.3;
                                    0.1 ,-0.3,2.1 ];

gmmParameters.covMatrices(:,:,2) = [1.7 ,-0.1 ,0.3 ;
                                    -0.1,2.3  ,-0.5;
                                    0.3 ,-0.5 ,1.5 ];
gmmParameters.covMatrices(:,:,3) = [3.0 ,-0.5,0.4 ;
                                    -0.5,2.8,-0.2;
                                    0.4 ,-0.2,3.2 ];
gmmParameters.covMatrices(:,:,4) = [2.9 ,-0.3,0.1 ;
                                    -0.3,3.1 ,-0.4;
                                    0.1 ,-0.4,2.7 ];

gmmParameters.covMatrices(:,:,1), 
gmmParameters.covMatrices(:,:,2), 
gmmParameters.covMatrices(:,:,3),
gmmParameters.covMatrices(:,:,4),

% Generate data from specified pdf
[x,labels] = generateDataFromGMM(N,gmmParameters); % Generate data
for l = 1:C
    Nclass(l,1) = length(find(labels==l));
end

% Evaluate class-conditional PDFs for each Gaussian component
pxgivenl = zeros(C, N); % Initialize
for l = 1:C
    if l == 3
        % Condition for L = 3 where it is a combination
        pxgivenl(l, :) = 0.5 * evalGaussianPDF(x, gmmParameters.meanVectors(:,3), gmmParameters.covMatrices(:,:,3)) + ...
                         0.5 * evalGaussianPDF(x, gmmParameters.meanVectors(:,4), gmmParameters.covMatrices(:,:,4));
    else
        % Class 1 and Class 2 each have one Gaussian
        pxgivenl(l, :) = evalGaussianPDF(x, gmmParameters.meanVectors(:,l), gmmParameters.covMatrices(:,:,l));
    end
end

% Calculate total probability
px = gmmParameters.priors*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,C,1); % P(L=l|x)

lossMatrix10 = [0,10,10;
                1,0 ,10;
                1,1 ,0];

lossMatrix = lossMatrix10;
expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
[~,decisions] = min(expectedRisks,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP

% The matrix computed here is P(D=d|L=l,x)
% It is called the confusion matrix if decisions are class labels.
hold on;
colors = {'g', 'r'}; % Green for correct, red for incorrect
shapes = {'o', 'x', 's'}; % Circle, X, square for each class
figure(9), clf,
for d = 1:C % each decision option
    for l = 1:C % each class label
        i_correct = find(decisions == d & labels == l);
        i_incorrect = find(decisions ~= d & labels == l);
        ConfusionMatrix(d,l) = length(i_correct)/length(find(labels==l));
        scatter3(x(1, i_correct), x(2, i_correct), x(3, i_correct), 36, colors{1}, shapes{l}, 'filled'); hold on, axis equal, grid on,
        scatter3(x(1, i_incorrect), x(2, i_incorrect), x(3, i_incorrect), 36, colors{2}, shapes{l}); hold on, axis equal, grid on,
    end
end

title('3D Scatter Plot of Classification Results');
xlabel('x'); ylabel('y'); zlabel('z');
legend({'Class 1 Correct', 'Class 1 Incorrect', 'Class 2 Correct', 'Class 2 Incorrect', 'Class 3 Correct', 'Class 3 Incorrect'});
grid on;
hold off;
ConfusionMatrix,

writematrix(x,'samples_2b_10.csv');
writematrix(labels,'samples_2b_10.csv','WriteMode','append');
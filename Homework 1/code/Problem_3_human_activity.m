% Expected risk minimization with C classes
clear all, close all,

% Loading Dataset
X1 = readmatrix('UCI HAR Dataset\train\X_train.txt');
X2 = readmatrix('UCI HAR Dataset\test\X_test.txt');

Y1 = readmatrix('UCI HAR Dataset\train\y_train.txt');
Y2 = readmatrix('UCI HAR Dataset\test\y_test.txt');


list_data = [X1;X2];
list_label = [Y1;Y2];

l1 = mean(list_data(:,1:90),2);
l2 = mean(list_data(:,91:180),2);
l3 = mean(list_data(:,181:270),2);
l4 = mean(list_data(:,271:360),2);
l5 = mean(list_data(:,361:450),2);
l6 = mean(list_data(:,451:end),2);

init_data = [l1,l2,l3,l4,l5,l6];

labels = list_label;

% Intializing Dataset

C = 6;
N = 50000; % Number of samples
n = 6; % Features for human activity

% Initialize priors vector
gmmParameters.priors = zeros(1, C); 

% Generates priors based on mean
for l = 1:C
    gmmParameters.priors(l) = sum(labels == l) / 10299; % Prior for each class
end

gmmParameters.meanVectors = zeros(1, C); 
for l = 1:C
    for m = 1:C
        gmmParameters.meanVectors(m,l) = sum(init_data(:,l)) / 10299;
    end
end

% Initialize covariance matrix
gmmParameters.covMatrices = zeros(n, n, C);

lambda = 0.75*trace(cov(init_data))/rank(cov(init_data));
C_regularized = cov(init_data) + lambda * eye(n,n);

for l = 1:C
    gmmParameters.covMatrices(:,:,l) = C_regularized;
end


% Generate data from specified pdf
[x,labels] = generateDataFromGMM(N,gmmParameters); % Generate data
for l = 1:C
    Nclass(l,1) = length(find(labels==l));
end

% Shared computation for both parts
for l = 1:C
    pxgivenl(l,:) = evalGaussianPDF(x,gmmParameters.meanVectors(:,l),gmmParameters.covMatrices(:,:,l)); % Evaluate p(x|L=l)
end

px = gmmParameters.priors*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,C,1); % P(L=l|x)


lossMatrix = [0,10,30,80,120,170;
              10,0,10,30,80,120;
              30,10,0,10,30,80;
              80,30,10,0,10,30;
              120,80,30,10,0,10;
              170,120,80,30,10,0];
lossMatrix = lossMatrix * 2;
expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
[~,decisions] = min(expectedRisks,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP

% It is called the confusion matrix if decisions are class labels.
colors = {'g', 'r'}; % Green for correct label, red for incorrect label
shapes = {'', '', 'o', '+', '*', '.', 'x', 'square', 'hexagram', '', ''}; % For each class that has prior > 0
figure(11), clf,
for d = 1:C % each decision option
    for l = 1:C % each class label
        i_correct = find(decisions == d & labels == l);
        i_incorrect = find(decisions ~= d & labels == l);
        ConfusionMatrix(d,l) = length(i_correct)/length(find(labels==l));
    end
end
ConfusionMatrix,

title('3D Scatter Plot of Classification Results');
xlabel('Citric Acid'); ylabel('pH'); zlabel('Alcohol');

count = 0;
for i = 1:length(decisions)
    if decisions(i) == labels(i)
        count = count + 1;
    end
end
count,

writematrix(x,'samples_3_human.csv');
writematrix(labels,'samples_3_human.csv','WriteMode','append');
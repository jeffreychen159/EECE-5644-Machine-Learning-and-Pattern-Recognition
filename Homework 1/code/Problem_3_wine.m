% Expected risk minimization with C classes
clear all, close all,

% Loading Dataset
data_wine = readtable('winedata.csv', 'Delimiter', ';');

% Intializing Dataset
init_data = table2array(data_wine(:, 1:end-1)); % Takes the features of the wine
labels = table2array(data_wine(:,end)); % Takes the last column

C = 11;
N = 10000; % Number of samples
n = 11; % Features for the wine

% Initialize priors vector
gmmParameters.priors = zeros(1, C); 

% Generates priors based on mean
for l = 1:C
    gmmParameters.priors(l) = sum(labels == l) / 4898; % Prior for each class
end

gmmParameters.meanVectors = zeros(1, C); 
for l = 1:C
    for m = 1:C
        gmmParameters.meanVectors(m,l) = sum(init_data(:,l)) / 4898;
    end
end

% Initialize covariance matrix
gmmParameters.covMatrices = zeros(n, n, C);

lambda = 0.75*trace(cov(init_data))/rank(cov(init_data));
C_regularized = cov(init_data) + lambda * eye(C,C);

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


lossMatrix = [0,5,10,15,20,25,30,35,40,45,50;
              5,0,5,10,15,20,25,30,35,40,45;
              10,5,0,5,10,15,20,25,30,35,40;
              15,10,5,0,5,10,15,20,25,30,35;
              20,15,10,5,0,5,10,15,20,25,30;
              25,20,15,10,5,0,5,10,15,20,25;
              30,25,20,15,10,5,0,5,10,15,20;
              35,30,25,20,15,10,5,0,5,10,15;
              40,35,30,25,20,15,10,5,0,5,10;
              45,40,35,30,25,20,15,10,5,0,5;
              50,45,40,35,30,25,20,15,10,5,0;];
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
        if (d > 1) && (d < 9)
            scatter3(x(3, i_incorrect), x(9, i_incorrect), x(11, i_incorrect), 36, colors{2}, shapes{l}); hold on, axis equal, grid on,
            scatter3(x(3, i_correct), x(9, i_correct), x(11, i_correct), 36, colors{1}, shapes{l}, 'filled'); hold on, axis equal, grid on,
        end
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

writematrix(x,'samples_3_wine.csv');
writematrix(labels,'samples_3_wine.csv','WriteMode','append');
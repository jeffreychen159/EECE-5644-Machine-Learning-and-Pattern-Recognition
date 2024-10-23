
%% P1 Part B

N = 10000; % number of iid samples
n = 4;
mu(:,1) = [-1;-1;-1;-1];
mu(:,2) = [1;1;1;1];
sigma(:,:,1) = [2,0,0,0;
                0,1,0,0;
                0,0,1,0;
                0,0,0,2];

sigma(:,:,2) = [1,0,0,0;
                0,2,0,0;
                0,0,1,0;
                0,0,0,3];
p = [0.35,0.65]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end
% Plotting Data
figure(3),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 
hold off; 

% Taken from LDAwithROCcurve
function [Pfp,Ptp,Perror,thresholdList] = ROCcurve(discriminantScores,labels)
[sortedScores,~] = sort(discriminantScores,'ascend');
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];
for i = 1:length(thresholdList)
    tau = thresholdList(i);
    decisions = (discriminantScores >= tau);
    Ptn(i) = length(find(decisions==0 & labels==0))/length(find(labels==0));
    Pfp(i) = length(find(decisions==1 & labels==0))/length(find(labels==0));
    Ptp(i) = length(find(decisions==1 & labels==1))/length(find(labels==1));
    Perror(i) = sum(decisions~=labels)/length(labels);
end
end

% ROC Curve
g1 = evalGaussian(x,mu(:,1),sigma(:,:,1)); g2 = evalGaussian(x,mu(:,2),sigma(:,:,2));
discriminantScoresERM = log(g2./g1);
figure(4),
[PfpERM,PtpERM,PerrorERM,thresholdListERM] = ROCcurve(discriminantScoresERM,label);

% Getting Gamma Experimental and Theoretical
% Typing to Console
gamma_theoretical_naive = 0.35/0.65; % Theoretical P(L=0) / P(L=1)
gamma_theoretical_naive, 

gamma_experimental_naive = Nc(1) / Nc(2);
gamma_experimental_naive, 

% Getting Labels
dis_0 = discriminantScoresERM(label == 0);
dis_1 = discriminantScoresERM(label == 1);

% Calculate lambdas
thy_lambda_0 = sum(dis_0 >= gamma_theoretical_naive) / length(dis_0); % False Positive Rate
thy_lambda_1 = sum(dis_1 >= gamma_theoretical_naive) / length(dis_1); % True Positive Rate
thy_lambdas = [thy_lambda_0, thy_lambda_1];

% Calculate probability of error
thy_p_err = thy_lambdas(1) * 0.7 + (1 - thy_lambdas(2)) * 0.3;

% Typing to Console
Perror_theoretical_naive = thy_p_err;
Perror_theoretical_naive,

Perror_experimental_naive = min(PerrorERM); 
Perror_experimental_naive, 

% Plotting ROC Curve
hold on, plot(PfpERM,PtpERM,'*m'),
hold on, plot(thy_lambda_0,thy_lambda_1,"*b"), 
xlabel('P(False+)'),ylabel('P(True+)'), title('ROC Curve'),

writematrix(x,'samples_1b.csv');
writematrix(label,'samples_1b.csv','WriteMode','append');
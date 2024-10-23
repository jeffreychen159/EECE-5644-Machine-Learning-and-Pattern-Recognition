clear all, close all,

% Generate n-dimensional data vectors from 2 Gaussian pdfs
n = 4; 
N1 = 5000; mu1 = [-1;-1;-1;-1]; A1 = [2,-0.5,0.3,0;-0.5,1,-0.5,0;0.3,-0.5,1,0;0,0,0,2];
N2 = 5000; mu2 = [1;1;1;1]; A2 = [1,0.3,-0.2,0;0.3,2,0.3,0;-0.2,0.3,1,0;0,0,0,3];
x1 = A1*randn(n,N1)+mu1*ones(1,N1);
x2 = A2*randn(n,N2)+mu2*ones(1,N2);
labels = [zeros(1,N1),ones(1,N2)];

% Estimate mean vectors and covariance matrices from samples
mu1hat = mean(x1,2); S1hat = cov(x1');
mu2hat = mean(x2,2); S2hat = cov(x2');

% Calculate the between/within-class scatter matrices
Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

% Solve for the Fisher LDA projection vector (in w)
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

% Linearly project the data from both categories on to w
y1 = w'*x1;
y2 = w'*x2;

% Plot the data before and after linear projection
figure(5),
hold on, subplot(2,1,1), plot(x1(1,:),x1(2,:),'r*');
hold on, plot(x2(1,:),x2(2,:),'bo'); axis equal,
hold on, subplot(2,1,2), plot(y1(1,:),zeros(1,N1),'r*');
hold on, plot(y2(1,:),zeros(1,N2),'bo'); axis equal,

% ROC curve for Fisher LDA
discriminantScoresLDA = [y1,y2];
[PfpLDA,PtpLDA,PerrorLDA,thresholdListLDA] = ROCcurve(discriminantScoresLDA,labels);
Perror_lda = min(PerrorLDA);
i = find(PerrorLDA == min(PerrorLDA));
figure(6),
hold on, plot(PfpLDA,PtpLDA,'*m'),
hold on, plot(PfpLDA(i),PtpLDA(i),'*b'),
xlabel('P(False+)'),ylabel('P(True+)'), title('ROC Curve for LDA Discriminant Scores'),
figure(7),
hold on, plot(thresholdListLDA,PerrorLDA,'og'),
xlabel('Thresholds'), ylabel('P(error) for LDA Discriminant Scores'),


function [Pfp,Ptp,Perror,thresholdList] = ROCcurve(discriminantScores,labels)
[sortedScores,ind] = sort(discriminantScores,'ascend');
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



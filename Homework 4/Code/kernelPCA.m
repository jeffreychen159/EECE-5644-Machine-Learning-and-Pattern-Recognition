function out = kernelPCA(N)
% Implements a basic Kernel PCA demo using Gaussian kernels
out = NaN;
close all,
%a = [zeros(1,N-4),poly([-1,0,1])]; 
sigmanoise = 1e-1;
x1dummy = rand(1,N);
x(1,:) = sort(x1dummy,'ascend');
%x(2,:) = a*vander(x(1,:)) + sigmanoise*randn(1,N);
x(2,:) = sqrt(1-x(1,:).^2)+sigmanoise*randn(1,N);
hue = linspace(0,1-1/N,N); sat = ones(1,N); val = ones(1,N);
colorlist = hsv2rgb([hue;sat;val]');
figure(1),
for i = 1:N
    % Using color (hue) as a visual cue
    plot(x(1,i),x(2,i),'.','color',colorlist(i,:)),
    hold on,
end

%K = 1;
sigma = 1e1;
pd = pdist2(x',x'); K = exp(-0.5*pd.^2/sigma^2);
[V,D] = eig(K);
[d,ind] = sort(diag(D),'descend');
V = V(:,ind); alpha = V(:,1);
%figure(2), stem(d,'.');

y = alpha'*K; % the ith column of K is k(xi)
figure(2), plot3(x(1,:),x(2,:),y(1,:),'.'),
xlabel('x_1'), ylabel('x_2'), zlabel('y_1'),

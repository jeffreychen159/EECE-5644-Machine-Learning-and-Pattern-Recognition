function g = evalGaussian(x,mu,Sigma)
% Evaluate the Gaussian pdf N(mu,Sigma) at each column of x
[n,N] = size(x);
C = ((2*pi)^n*det(Sigma))^(-1/2); % normalization constant
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1); % exponent
g = C*exp(E);% Gaussian PDF values in a 1xN row vector

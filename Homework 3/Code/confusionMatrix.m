function [confusionMatrix,classPriors,expectedRisk] = confusionMatrix(labels,decisions)
% assumes labels are in {1,...,L} and decisions are in {1,...,D}
L = length(unique(labels));
D = length(unique(decisions));
confusionMatrix = zeros(D,L);
for l = 1:L
    Nl = length(find(labels==l));
    for d = 1:D
        Ndl = length(find(labels==l & decisions==d));
        confusionMatrix(d,l) = Ndl/Nl;
    end
    classPriors(l,1) = Nl/length(labels); % class prior for label l
end
if L==D
    pCorrrect = sum(diag(confusionMatrix).*classPriors);
    pError = 1-pCorrect;
end
expectedRisk = sum(sum(lossMatrix.*confusionMatrix.*repmat(classPriors,1,L),2),1);
clc;clear
tic
interaction = load(["./smallrawdata1/Association matrix.txt"]);
disease_ss  = load(["./smallrawdata1/disease semantic similarity.txt"]);
%Êý¾Ý¼¯2
% interaction=importdata('./smallrawdata2/chr_diseasematrix.csv'); 
% disease_ss=csvread('./smallrawdata2/dissimilarity.csv',1);

[nc,nd ]= size(interaction);
ciRNA_ss = miRNASS( interaction, disease_ss ); % Finding ciRNA functional similarity
maxiter = 20;

restartProb = 0.7;
[CC,DD]=gkl(nc,nd,interaction);
circ_feature = zeros(nc,nc);

for i = 1:nc
    for j = 1:nc
        if ciRNA_ss(i,j)~=0
            circ_feature(i,j) = ciRNA_ss(i,j); % Functional
        else 
            circ_feature(i,j) = CC(i,j); %Gaussian
        end
    end
end

dis_feature = zeros(nd,nd);

for i = 1:nd
    for j = 1:nd
        if disease_ss(i,j)~=0
            dis_feature(i,j) = disease_ss(i,j); % Functional
        else 
            dis_feature(i,j) = DD(i,j);  % Gaussian
        end
    end
end

%
dlmwrite('./smalldata1/integrated circRNA similarity.txt',circ_feature,'delimiter','\t');
dlmwrite('./smalldata1/integrated disease similarity.txt',dis_feature,'delimiter','\t');
% 
% RWR
tic
X =diffusionRWR(dis_feature, restartProb, maxiter);
nnode = size(X, 1);
alpha = 1 / nnode;
X = log(X + alpha) - log(alpha);
toc
tic
Y = diffusionRWR(circ_feature, restartProb, maxiter);
nnode = size(Y, 1);
alpha = 1 / nnode;
Y = log(Y + alpha) - log(alpha);
toc

dlmwrite(['./smallfeaturedata1/diseaseNets.txt'], X, '\t');
dlmwrite(['./smallfeaturedata1/circRNANets.txt'], Y, '\t');



X = load(["./smallfeaturedata1/diseaseNets.txt"]);
Y = load(["./smallfeaturedata1/circRNANets.txt"]);
%PCA 0.8 0.8
dis_endfeature=PCA(X,0.8);
circ_endfeature=PCA(Y,0.8);

dlmwrite(['./smallenddata1/diseaseNets.txt'], dis_endfeature, '\t');
dlmwrite(['./smallenddata1/circRNANets.txt'], circ_endfeature, '\t');
toc

       

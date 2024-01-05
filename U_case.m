clc;clear
%  调试操作
% 
interaction = load([".\smallrawdata1\Association matrix.txt"]);
disease_feat = load('./smallenddata1/diseaseNets.txt');
circRNA_feat = load('./smallenddata1/circRNANets.txt');
interaction=interaction'
interaction1=interaction
seed = 2;
rng(seed);
shape_train=[];
shape_test=[];
Pint = find(interaction1); 
Nint = length(Pint); 
Pnoint = find(~interaction);
Pnointnpart = Pnoint(randperm(length(Pnoint), Nint *1));
Nnoint = length(Pnoint); 
Pint = Pint(randperm(length(Pint), Nint * 1));

train_idx = [Pint; Pnointnpart];
[I, J] = ind2sub(size(interaction), train_idx);
B=disease_feat(I,:);
C=circRNA_feat(J,:);
trainint=[B,C];

Ytrain = [ones(length(Pint), 1); zeros(length(Pnointnpart), 1)];
shape_train=[shape_train,size(Ytrain,1)];

test_idx=[Pnoint]
[k,z] = ind2sub(size(interaction), test_idx);
D = disease_feat(k,:);
F = circRNA_feat(z,:);
testint=[D,F];
Ytest = [zeros(length(test_idx), 1)];
shape_test=[shape_test,size(testint,1)];

abc=0;
dlmwrite(['./Ucase/testlabel/testlabel', num2str(abc), '.txt'],Ytest, '\t');
dlmwrite(['./Ucase/train/train', num2str(abc), '.txt'],trainint, '\t');
dlmwrite(['./Ucase/test/test', num2str(abc), '.txt'],testint, '\t');
dlmwrite(['./Ucase/trainlabel/trainlabel', num2str(abc), '.txt'],Ytrain, '\t');
 
dlmwrite(['./Ucase/trainrow.txt'],shape_train, '\t');
dlmwrite(['./Ucase/testrow.txt'],shape_test, '\t');

%  调试暂停，在python中运行
% 导入prob.txt文件

 Ytest=load(["./Ucase/prob.txt"]);
 Scores=Ytest;
%  
 
 [~,ind] = sort(Scores,'descend');
 for i=1:50830
       interaction(k(i),z(i))=Ytest(i);
end
i=1;
x=zeros(30,1);
y=zeros(30,1);
v=zeros(30,1);
 %    interaction(k(ind(i)),z(ind(i)))=Ytest(ind(i));
while i<51
    x(i)=k(ind(i));
    y(i)=z(ind(i));
    v(i)=Ytest(ind(i));
    i=i+1;
end
%  %dlmwrite(['./data2/teslabelt', '.txt'],Ytest, '\t');
% % dlmwrite(['./data2/train', '.txt'],trainint, '\t');
%  %dlmwrite(['./data2/test', '.txt'],testint, '\t');
%  %dlmwrite(['./data1/trainlabel', '.txt'],Ytrain, '\t');
 dlmwrite(['./data/interaction2', '.txt'],interaction', '\t');
% 


       


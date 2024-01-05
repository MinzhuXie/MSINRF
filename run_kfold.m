clear all
clc
tic

% 
interaction = load(["./smallrawdata1/Association matrix.txt"]);
disease_feat = load('./smallenddata1/diseaseNets.txt');
circRNA_feat = load('./smallenddata1/circRNANets.txt');

interaction=interaction'
nFold = 5;
shape_train=[];
shape_test=[];
seed = 2;

rng(seed);
Pint = find(interaction); 
Nint = length(Pint);   
Pnoint = find(~interaction);
Pnoint = Pnoint(randperm(length(Pnoint), Nint * 1));
Nnoint = length(Pnoint); 
Pint = Pint(randperm(length(Pint), Nint * 1));
posFilt = crossvalind('Kfold', Nint, nFold);
negFilt = crossvalind('Kfold', Nnoint, nFold);
	for foldID = 1 : nFold
        i=foldID;
        abc=i-1;
		train_posIdx = Pint(posFilt ~= foldID);
		train_negIdx = Pnoint(negFilt ~= foldID);
		train_idx = [train_posIdx; train_negIdx];
		Ytrain = [ones(length(train_posIdx), 1); zeros(length(train_negIdx), 1)];
        
        shape_train=[shape_train,size(Ytrain,1)];

		test_posIdx =Pint(posFilt == foldID);
		test_negIdx = Pnoint(negFilt == foldID);
		test_idx = [test_posIdx; test_negIdx]; 
		Ytest = [ones(length(test_posIdx), 1); zeros(length(test_negIdx), 1)]; 

		[I, J] = ind2sub(size(interaction), train_idx);
        
        B=disease_feat(I,:);
        C=circRNA_feat(J,:);
        trainint=[B,C];
        [k,z] = ind2sub(size(interaction), test_idx);
        D = disease_feat(k,:);
        F = circRNA_feat(z,:);
        testint=[D,F];
        testindex=[k,z];
        shape_test=[shape_test,size(testint,1)];
        
        dlmwrite(['./cnn_input/testlabel/testlabel', num2str(abc), '.txt'],Ytest, '\t');
        dlmwrite(['./cnn_input/train/train', num2str(abc), '.txt'],trainint, '\t');
        dlmwrite(['./cnn_input/test/test', num2str(abc), '.txt'],testint, '\t');
        dlmwrite(['./cnn_input/trainlabel/trainlabel', num2str(abc), '.txt'],Ytrain, '\t');

     end
dlmwrite(['./cnn_input/trainrow.txt'],shape_train, '\t');
dlmwrite(['./cnn_input/testrow.txt'],shape_test, '\t');
toc

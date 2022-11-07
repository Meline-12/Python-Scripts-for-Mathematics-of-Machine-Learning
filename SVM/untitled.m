S = load('data_KSVM.mat');
csvwrite('data_KSVM_X.csv', S.x);
csvwrite('data_KSVM_Y.csv', S.y);
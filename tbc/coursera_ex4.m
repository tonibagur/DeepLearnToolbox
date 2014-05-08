

%% Loading data
fprintf('Loading and Visualizing Data ...\n')
load('ex4data1.mat');

sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));
%%y(sel) %Uncomment to see the labels of visualized data
labels=[1 2 3 4 5 6 7 8 9 10]
yvec=double(repmat(labels,size(y,1),1)==repmat(y,1,size(labels,2)));
%%yvec(sel,:)

[xtrain,ytrain,xtest,ytest,xval,yval]=get_train_test_val(X,yvec);

iters=10
errors=zeros(iters,4);
er_min=1000
lambda_min=-1
for lambda=1:iters

    nn = nnsetup([400 120 10]);
    nn.activation_function = 'sigm';    %  Sigmoid activation function
    %%%%nn.learningRate = 1;                %  Sigm require a lower learning rate ---> 
    nn.output='softmax';
    nn.weightPenaltyL2=-(lambda-1)*1000;
    %%%%errors(lambda,3)=nn.weightPenaltyL2;
    errors(lambda,4)=lambda
    opts.numepochs =  1;   %  Number of full sweeps through data
    opts.batchsize = 50;  %  Take a mean gradient step over this many samples
    opts.plot=1;
    [nn, L] = nntrain(nn, xtrain, ytrain, opts);
    [er, bad] = nntest(nn, xtrain, ytrain);
    errors(lambda,1)=er;
    [er, bad] = nntest(nn, xval, yval);
    errors(lambda,2)=er;
    if er< er_min
       lambda_min=lambda
       er_min=er
    end
    [er, bad] = nntest(nn, xtest, ytest);
    errors(lambda,3)=er;

end

pause
lambda_min
errors

%%displayData(nn.W{1}(:, 2:end));

%%load mnist_uint8;
%%displayData(train_x(sel,:));
%%train_y(sel)






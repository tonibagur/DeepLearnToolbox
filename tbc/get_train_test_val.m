function [xtrain,ytrain,xtest,ytest,xval,yval]=get_train_test_val(x,y)

   m=size(x,1);
   n=size(x,2);
   perm=randperm(m);
   bound1=floor(m*0.6);
   bound2=floor(m*0.8);
   xtrain=x(perm(1:bound1),:);
   ytrain=y(perm(1:bound1),:);
   xtest=x(perm(bound1+1:bound2),:);
   ytest=y(perm(bound1+1:bound2),:);
   xval=x(perm(bound2+1:m),:);
   yval=y(perm(bound2+1:m),:);
end

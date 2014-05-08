function [X,y,files]=get_chess_data()
    files_p=glob('/home/coneptum/chessvision/images/positives/*.png');
    X=[];
    y=[];
    files=[];
    for i = 1:length(files_p)
        f=files_p{i,1};
        files=[files;f];
        im=imread(f)(:);
        #im=im + ones(size(im));
        X=[X;im'];
        y=[y;[1]];
    end
    files_p=glob('/home/coneptum/chessvision/images/negatives/*.png');
    for i = 1:length(files_p)
        f=files_p{i,1};
        files=[files;f];
        im=imread(f)(:);
        #im=im + ones(size(im))
        X=[X;im'];
        y=[y;[2]];
    end
    X=double(X);
    y=double(y);

#! /usr/bin/octave --persist

data=load("-ascii","tr.dat");
labels=load("-ascii","trlabels.dat");
nc = 4;

for i = 1:nc
    for j = i+1:nc
        di = data(labels==i,:); li = labels(labels==i);
        dj = data(labels==j,:); lj = labels(labels==j);
        d = [di;dj]; l = [li;lj];
        A(i,j) = svmtrain(l,d,"-t 0 -c 1000");
    endfor
endfor

A

#! /usr/bin/octave --persist


nc = 4;

#######################################################
#	Entrenamiento
#######################################################

data=load("-ascii","tr.dat");
labels=load("-ascii","trlabels.dat");


for i = 0:nc-1
    for j = i+1:nc-1
        di = data(labels==i,:); li = labels(labels==i);
        dj = data(labels==j,:); lj = labels(labels==j);
        d = [di;dj]; l = [li;lj];
        svm(i+1,j+1) = svmtrain(l,d,"-t 0 -c 1000 -q");
    endfor
endfor

svm


#######################################################
#	Test
#######################################################

data=load("-ascii","ts.dat");
labels=load("-ascii","tslabels.dat");
t = [];
l = [];
for i = 0:nc-1
    di = data(labels==i,:); li = labels(labels==i);
    t = [t;di]; l = [l;li];
endfor

pred14 = svmpredict(l,t,svm(1,4),"");
i141 = find(pred14 == 0);
i144 = find(pred14 == 3);

pred24 = svmpredict(l(i144),t(i144),svm(2,4),"");
i242 = find(pred24 == 1);
i244 = find(pred24 == 3);
pred13 = svmpredict(l(i141),t(i141),svm(1,3),"");
i131 = find(pred13 == 0);
i133 = find(pred13 == 2);

pred34 = svmpredict(l(i144(i244)),t(i144(i244)),svm(3,4),"");
i343 = find(pred34 == 2);
i344 = find(pred34 == 3);
pred(i144(i244(i343))) = 2;
pred(i144(i244(i344))) = 3;

pred23 = svmpredict(l(i144(i242)),t(i144(i242)),svm(2,3),"");
i232 = find(pred23 == 1);
i233 = find(pred23 == 2);
pred(i144(i242(i232))) = 1;
pred(i144(i242(i233))) = 2;

pred23 = svmpredict(l(i141(i133)),t(i141(i133)),svm(2,3),"");
i232 = find(pred23 == 1);
i233 = find(pred23 == 2);
pred(i141(i133(i232))) = 1;
pred(i141(i133(i233))) = 2;

pred12 = svmpredict(labels(i141(i131)),data(i141(i131)),svm(1,2),"");
i121 = find(pred12 == 0);
i122 = find(pred12 == 1);
pred(i141(i131(i121))) = 0;
pred(i141(i131(i122))) = 1;

pred = pred';



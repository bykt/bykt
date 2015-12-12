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
		eval(sprintf("train_%d_%d = svmtrain(l,d,\"-t 0 -c 1000 -q\");",i,j));
    endfor
endfor


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

for i = 0:nc-1
    for j = i+1:nc-1
		eval(sprintf("predict_%d_%d = svmpredict(l,t,train_%d_%d,\"\");",i,j,i,j));
    endfor
endfor







#! /usr/bin/octave --persist


data=load("-ascii","tr.dat");
labels=load("-ascii","trlabels.dat");
res=svmtrain(labels,data,"-t 0 -c 1000");
res

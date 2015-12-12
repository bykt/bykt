#! /usr/bin/octave --persist


global nc = 4;

#######################################################
#	Entrenamiento
#######################################################

data=load("-ascii","tr.dat");
labels=load("-ascii","trlabels.dat");

# Generamos los SVM posibles
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

# Seleccionamos solamente las muestras de las clases de 0 a nc-1
for i = 0:nc-1
    di = data(labels==i,:); li = labels(labels==i);
    t = [t;di]; l = [l;li];
endfor

# Predecimos cada muestra para cada SVM generado
for i = 0:nc-1
    for j = i+1:nc-1
		eval(sprintf("global predict_%d_%d = svmpredict(l,t,train_%d_%d,\"-q\");",i,j,i,j));
    endfor
endfor


#######################################################
#	DAG
#######################################################

function[p] = dag(m,c1,c2,depth)
	global nc;
	eval(sprintf("global predict_%d_%d",c1,c2));
	cond = eval(sprintf("predict_%d_%d (m) == c1",c1,c2));
	if(depth == nc-2)
		if(cond)
			p = c1;
		else
			p = c2;
		endif
	else
		if(cond)
			p = dag(m,c1,--c2,++depth);
		else
			p = dag(m,++c1,c2,++depth);
		endif
	endif
end

for i = 1:rows(t)
	predict(i) = dag(i,0,nc-1,0);
endfor

predict = predict';



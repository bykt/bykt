#! /usr/bin/octave --persist


nc = 4;

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
		eval(sprintf("predict_%d_%d = svmpredict(l,t,train_%d_%d,\"\");",i,j,i,j));
    endfor
endfor


#######################################################
#	DAG
#######################################################

function[p] = dag(m,c1,c2,depth)
	A = 6
	if(depth == nc-2)
		eval(sprintf("if(predict_%d_%d(m) == c1)",c1,c2));
		#if(predict_c1_c2(m) == c1)
			p = c1;
		else
			p = c2;
		eval(sprintf("endif"));
	else
		A = 5
		eval(sprintf("if(predict_%d_%d(m) == c1)",c1,c2));
			p = dag(m,c1,--c2,++depth);
		else
			p = dag(m,++c1,c2,++depth);
		eval(sprintf("endif"));
	endif
end

for i = 1:rows(t)
	predict(i) = dag(i,0,nc-1,0)
endfor

predict = predict'



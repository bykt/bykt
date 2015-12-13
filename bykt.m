#! /usr/bin/octave --persist


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
endfunction


function multi(tr,trl,ts,tsl,C,kernel)

	global nc;

	############################################################################################
	#	Entrenamiento
	############################################################################################

	data=load("-ascii",tr);
	labels=load("-ascii",trl);

	# Generamos los SVM posibles
	for i = 0:nc-1
		for j = i+1:nc-1
		    di = data(labels==i,:); li = labels(labels==i);
		    dj = data(labels==j,:); lj = labels(labels==j);
		    d = [di;dj]; l = [li;lj];
			eval(sprintf("train_%d_%d = svmtrain(l,d,\"-t %s -c %f -q\");",i,j,kernel,C));
		endfor
	endfor


	############################################################################################
	#	Test
	############################################################################################

	data=load("-ascii",ts);
	labels=load("-ascii",tsl);

	t = [];
	l = [];

	# Seleccionamos solamente las muestras de las clases de 0 a nc-1
	for i = 0:nc-1
		di = data(labels==i,:); li = labels(labels==i);
		t = [t;di]; l = [l;li];
	endfor

	# Predecimos todas las muestras para cada SVM generado
	for i = 0:nc-1
		for j = i+1:nc-1
			eval(sprintf("global predict_%d_%d = svmpredict(l,t,train_%d_%d,\"-q\");",i,j,i,j));
		endfor
	endfor


	############################################################################################
	#	Votacion
	############################################################################################

	rand("seed",10);

	for k = 1:rows(t)
		ac = zeros(1,nc);
		for i = 0:nc-1
			for j = i+1:nc-1
				class = eval(sprintf("predict_%d_%d(k)",i,j));
				++ac(class+1);
			endfor
		endfor
		[v,winner] = max(ac);
		all_max = find(ac == v);
		if(columns(all_max) == 2)l
			winner = eval(sprintf("predict_%d_%d(k)",all_max(1)-1,all_max(2)-1));
		else
			if(columns(all_max) > 2)
				rand_max = floor(rand(1)*columns(all_max) + 1);
				winner = all_max(rand_max);
			endif
		endif
		predict_vot(k) = winner-1;
	endfor

	predict_vot = predict_vot';


	############################################################################################
	#	DAG
	############################################################################################

	for k = 1:rows(t)
		predict_dag(k) = dag(k,0,nc-1,0);
	endfor

	predict_dag = predict_dag';

	comp = [l predict_vot predict_dag];

endfunction

allC = [0.01 0.1 1 10 100];
global nc = 4;

for p = 1:columns(allC)
	allC(p)
	multi("tr.dat","trlabels.dat","ts.dat","tslabels.dat",allC(p),"0");allC(p)
	multi("tr.dat","trlabels.dat","ts.dat","tslabels.dat",allC(p),"1 -d 2");allC(p)
	multi("tr.dat","trlabels.dat","ts.dat","tslabels.dat",allC(p),"1 -d 3");allC(p)
	multi("tr.dat","trlabels.dat","ts.dat","tslabels.dat",allC(p),"2");
endfor



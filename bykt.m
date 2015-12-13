#! /usr/bin/octave --persist


function multi(tr,trl,ts,tsl,nc,C,kernel)

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
			eval(sprintf("predict_%d_%d = svmpredict(l,t,train_%d_%d,\"-q\");",i,j,i,j));
		endfor
	endfor


	############################################################################################
	#	Votacion
	############################################################################################

	rand("seed",10); # Partimos siempre de la misma semilla

	# Por cada muestra
	for k = 1:rows(t)
		ac = zeros(1,nc);
		# Hacemos la votacion
		for i = 0:nc-1
			for j = i+1:nc-1
				class = eval(sprintf("predict_%d_%d(k)",i,j));
				++ac(class+1);
			endfor
		endfor

		# Escogemos el ganador
		[v,winner] = max(ac);
		all_max = find(ac == v);

		# Si hay empate entre 2, se escoge la clase que gane al aplicar el svm de esas 2 clases
		if(columns(all_max) == 2)
			winner = eval(sprintf("predict_%d_%d(k)",all_max(1)-1,all_max(2)-1));

		# Si hay empate entre mas de 2, se escoge una clase al azar
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
	
	# Por cada muestra
	for k = 1:rows(t)
		c1 = 0;
		c2 = nc-1;
		depth = 0;

		# Hacemos uso del arbol para determinar la clase
		while(depth < nc-2)
			cond = eval(sprintf("predict_%d_%d (k) == c1",c1,c2));
			if(cond)
				--c2;
			else
				++c1;
			endif
			++depth;
		endwhile
		cond = eval(sprintf("predict_%d_%d (k) == c1",c1,c2));
		if(cond)
			predict_dag(k) = c1;
		else
			predict_dag(k) = c2;
		endif
	endfor

	predict_dag = predict_dag';


	############################################################################################
	#	Evaluacion
	############################################################################################
	
	switch (kernel)
  		case "0"
    		used_k = "lineal";
 		case "1 -d 2"
    		used_k = "polinomial d=2";
		case "1 -d 3"
    		used_k = "polinomial d=3";
		case "2"
    		used_k = "radial";
  		otherwise
    		used_k = "sin concretar";
	endswitch

	comp = l == predict_vot;
	fallos = find(comp == 0);
	error = rows(fallos)/rows(comp) * 100;
	out = sprintf("Error usando votacion con kernel %s: %f ",used_k,error);
	out = [out "%"];
	disp(out);
	comp = l == predict_dag;
	fallos = find(comp == 0);
	error = rows(fallos)/rows(comp) * 100;
	out = sprintf("Error usando DAG con kernel %s: %f ",used_k,error);
	out = [out "%"];
	disp(out);
	disp("----------");
	
endfunction


############################################################################################
#	MAIN
############################################################################################

nc = 4;
allC = [0.01 0.1 1 10 100];

for p = 1:columns(allC)
	disp(sprintf("Utilizando C = %f",allC(p)));
	multi("tr.dat","trlabels.dat","ts.dat","tslabels.dat",nc,allC(p),"0");
	multi("tr.dat","trlabels.dat","ts.dat","tslabels.dat",nc,allC(p),"1 -d 2");
	multi("tr.dat","trlabels.dat","ts.dat","tslabels.dat",nc,allC(p),"1 -d 3");
	multi("tr.dat","trlabels.dat","ts.dat","tslabels.dat",nc,allC(p),"2");
	disp("************************************************************");
endfor



#! /usr/bin/octave --persist

function[] = recursivo(i)
	i
	++i
	i
	recursivo(i);
end

recursivo(0);

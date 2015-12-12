#! /usr/bin/octave --persist
global a = 1;

function[b] = dag(b)
	global a;
	a
end

dag(4);

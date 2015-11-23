###############
#########
#testing script


n = 50;
n_var = 0; 
avg_var = 0; 
while(true)
	noise = randn(n);
	var_noise = var(noise);
	avg_var = avg_var + var_noise;
	n_var = n_var + 1;
	#avg_var = avg_var + var_noise*(n-1);
	#n_var = n_var + n;
	println(avg_var/n_var);
end
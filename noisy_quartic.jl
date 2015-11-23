using Distributions
using Convex
using ECOS
using PyPlot

solver = ECOSSolver(verbose = false)
set_default_solver(solver);
d = Normal(0,0.1);

function grad(x)
	return 4*x^3;
end

function stochastic_grad(x, numSamples)
	avg_grad = 0; 
	if(numSamples == -1)
		noise = 0; 
	else
		noise = mean(rand(d,numSamples));
	end
	#noise = 0;
	return grad(x) + noise; 
end


############################################################

##### Figure 1 Noise-free Quartic ###############################
println("Noise-free quartic figure 1");

############################################################

function SGD(x_0, alpha, tol, grad_fun)
	while(norm(grad_fun(x_0)) > tol)
		x_0 = x_0 - alpha*grad_fun(x_0);
	end
	return abs(x_0);
end

function TSGD(x_0, alpha, tol, grad_fun)
	while(norm(grad_fun(x_0)) > tol)
		x_0 = x_0 - alpha*grad_fun(x_0)/max(1, alpha*norm(grad_fun(x_0)));
		#x_0 = x_0 - alpha*grad_fun(x_0)/(1 + alpha*norm(grad_fun(x_0)));
		#println(x_0);
	end
	return abs(x_0);
end

tol_array = [10.0^(-5), 10.0^(-6), 10.0^(-7), 10.0^(-8)]
col_array = ["b","g","r", "c"];

#=

col_ind = 1; 
for tol in tol_array
	a_vec = 0.1*2.0.^(-(0:5));
	x0_vec = [1 ,10, 50];
	mark_array = ["x", ">", "o"];
	ind = 1; 
	tic();
	for x_0 in x0_vec
		println(string("doing: ",x_0));
		sgd_line = Float64[];
		tsgd_line = Float64[];
		for alpha in a_vec
			push!(tsgd_line,TSGD(x_0, alpha, tol, grad));
			push!(sgd_line, SGD(x_0, alpha, tol, grad))
		end
		loglog(a_vec, tsgd_line, label = string("TSGD: \$x_0\$=",x_0, " tol=", tol), marker = mark_array[ind], color = col_array[col_ind]);
		ind = ind+1;
		loglog(a_vec, sgd_line, label = string("SGD: \$x_0\$=",x_0), marker = "<");
	end
	col_ind = col_ind +1; 
	toc();
end

ax = gca() # Get the handle of the current axis

#shrink current axis by 30% horiz 
box = ax[:get_position]();
ax[:set_position]([box[:x0], box[:y0],box[:width]*0.7,box[:height]]);

# Set the y axis to a logarithmic scale
ax[:set_ylim]([10.0^(-3),10.0^(-1)]);

#put legend to right of current axis
legend(fancybox="true", loc = "center left", fontsize = 11,bbox_to_anchor=(1, 0.5));

title(string("Noise-free quartic"));
xlabel("\$\\alpha\$");
ylabel("absolute error");

savefig(string("../figures/quartic_noisefree.png"));
=#

############################################################

##### Figure 2 Noisy Quartic ###############################
println("Noisy quartic figure 2")

############################################################



curr_x = 500;
temp_x = 9;
##step size
epsilon = 0.001;
test_1 = false;

alpha_array = [0.1, 0.05, 0.025, 0.001];

ind = 1; 
for alpha in alpha_array
	numSamples = 1; 
	numSampleRuns = 100; 
	numSteps = 2000;
	avg = zeros(numSteps);
	for i = 1:numSampleRuns
		curr_x = 500;
		current_xs = Float64[];
		for i = 1:numSteps
			temp_x = curr_x;
			if(test_1)
				curr_x = temp_x - alpha*stochastic_grad(temp_x,numSamples);
			else
				SG = stochastic_grad(temp_x,numSamples);
				curr_x = temp_x - alpha*SG/max(1, alpha*abs(SG));
			end
			#println(i);
			push!(current_xs,curr_x);
		end
		semilogy(1:numSteps, abs(current_xs), linestyle = "--", linewidth = 0.5);
		avg = avg + current_xs;
	end
	semilogy(1:numSteps, avg/numSampleRuns, linewidth = 2, color = "black");
	ax = gca() # Get the handle of the current axis
	ax[:set_ylim]([10.0^(-6),10.0^(3)]);# Set the y axis to a logarithmic scale
	title(string("TSGD alpha=",alpha));
	xlabel("k");
	ylabel("|x_k|");
	println(ind);
	savefig(string("../figures/quartic_n", numSamples, "_alpha", ind, ".png"));
	ind = ind+1;
	clf();
end

############################################################

##### Figure 3 Noisy Quartic ###############################
println("Noisy quartic figure 3")

############################################################
#=
###add variance stuff and confidence intervals...

function TSGD(x_0, alpha, tol, grad_fun, numSamples)
	SG = grad_fun(x_0,numSamples);
	niters = 1;
	while(norm(SG) > tol)
		x_0 = x_0 - alpha*SG/(alpha*norm(SG) + 1);
		SG = grad_fun(x_0,numSamples);
		niters = niters + 1;
	end
	return [x_0, niters];
end

##calculate sample mean over varying n starting at x_0 = 10
x_0 = 10; 
tol = 1e-5;
alpha = 0.01;


n_samples_vec = [1,10,1000,-1];
N_vec = logspace(1,2.5,10);


for n in n_samples_vec
	println(n);
	mean_xstar = Float64[];
	var_xstar = Float64[];
	for N in N_vec
		mean_vec = Float64[];
		for i = 1:N
			(xstar, niters) = TSGD(x_0, alpha, tol, stochastic_grad, n);
			push!(mean_vec, xstar)
		end
		push!(mean_xstar, mean(mean_vec));
		push!(var_xstar, var(mean_vec));
	end
	sigm_xstar = sqrt(var_xstar);
	#er_xstar = mean_xstar;
	er_xstar = abs(mean_xstar);
	println(sigm_xstar);
	if(n == -1)
		errorbar(N_vec, er_xstar, sigm_xstar.*1.96./n, label = string("noiseless ", n));
	else
		errorbar(N_vec, er_xstar, sigm_xstar.*1.96./n, label = string("n= ", n));
	end
end
legend(fancybox="true")
xlabel("N");
ylabel("error");
ax = gca();
ax[:set_xscale]("log");
ax[:set_yscale]("log");
title(string("\$\\alpha\$=",alpha));
tol = string(tol);
tol = replace(tol, ".0", "");
savefig(string("../figures/noisy_quartic_vartol=",tol,".png"));
=#

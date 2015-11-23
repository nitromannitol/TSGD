using Distributions
using Convex
using ECOS
using PyPlot

function grad(vector)
	x = vector[1];
	y = vector[2];
	return [400*x^3 - 400*x*y + 2*x - 2; 200*y - 200*x^2];
end

function stochastic_grad(vector, numSamples)
	d = Normal(0,0.1);
	avg_grad = 0; 
	for z in 1:numSamples;
		noise = rand(d,1)[1];
		avg_grad = avg_grad + grad(vector) + noise
	end
	return avg_grad/numSamples;
end


function SGD(x_0, alpha, tol, grad_fun, numSamples)
	niter = 1;
	SG = grad_fun(x_0,numSamples);
	x_0 = [x_0[1], x_0[2]];
	while(norm(SG) > tol && niter < 10^6)
		x_0 = x_0 - alpha*SG;
		niter = niter + 1; 
		SG = grad_fun(x_0,numSamples);
		if(isinf(x_0[1]) || isinf(x_0[2])) break; end
		if(isnan(x_0[1]) || isnan(x_0[2])) break; end
	end
	#println(string("SGD: ", niter));
	error = norm(x_0 - [1,1])^2;
	#if(error > 10)
	#	error = 5; 
	#end
	return error, niter;
end

function TSGD(x_0, alpha, tol, grad_fun, numSamples)
	SG = grad_fun(x_0,numSamples);
	x_0 = [x_0[1], x_0[2]];
	niter = 1;
	while(norm(SG) > tol && niter < 10^7)
		x_0 = x_0 - alpha*SG/max(alpha*norm(SG),1);
		SG = grad_fun(x_0,numSamples);
		niter = niter + 1;
	end
	#println(string("TSGD: ", niter));
	error = norm(x_0 - [1,1])^2;
	if(error > 10)
		error = 10; 
	end
	return error, niter;
end


numSamples = 1; 

############################################################

##### Figure 1 Noisey Rosenbrock ###############################
println("Noisy Rosenbrock Figure 1");

############################################################
#=

for tol in [10.0^(-1), 10.0^(-2), 10.0^(-3), 10.0^(-4)]
	a_vec = logspace(-5,0,15);
	x0_vec = [(0,0), (0.5,0.5), (2,2)];
	tic();
	for x_0 in x0_vec
		#take average over numAvgs iterations
		ss = size(a_vec,1);
		sgd_line = zeros(ss);
		tsgd_line = zeros(ss);
		numAvgs = 5; 
		for z = 1:numAvgs
			#println(z);
			for k in 1:(size(a_vec,1))
				alpha = a_vec[k];
				tsgd_line[k] = tsgd_line[k] + TSGD(x_0, alpha, tol, stochastic_grad, numSamples);
				sgd_line[k] = sgd_line[k] + SGD(x_0, alpha, tol, stochastic_grad,numSamples);
			end
		end
		sgd_line = sgd_line./numAvgs;
		tsgd_line = tsgd_line./numAvgs;

		loglog(a_vec, tsgd_line, label = string("TSGD: \$x_0\$=",x_0), marker = "o");
		loglog(a_vec, sgd_line, label = string("SGD: \$x_0\$=",x_0), marker = "x");

	end
	toc();
	legend(loc = "lower right", fancybox="true")
	ax = gca() # Get the handle of the current axis
	ax[:set_ylim]([10.0^(-5), 10.0^(1.1)]);# Set the y axis to a logarithmic scale
	title(string("Noisy Rosenbrock tol=",tol));
	xlabel("\$\\alpha\$");
	ylabel("distance from true minimum");
	tol = string(tol);
	tol = replace(tol, ".0", "");
	savefig(string("../figures/noisyrosenbrock_tol",tol,".png"));
	clf();
end
=#
############################################################

##### Table 1 Noisey Rosenbrock ###############################
println("Noisy Rosenbrock Table 1");

############################################################
#=
samplePaths = 500; 
TSGD_error, TSGD_niter = 0,0;
SGD_error, SGD_niter = 0,0; 
tol = 1e-1;
alpha = 0.002;
x_0 = (1.89,1.89)
for i = 1:samplePaths
	error,niter =  TSGD(x_0, alpha, tol, stochastic_grad, 1);
	TSGD_error = TSGD_error + error; 
	TSGD_niter = TSGD_niter + niter;
	error, niter = SGD(x_0, alpha, tol, stochastic_grad, 1);
	SGD_error = SGD_error + error; 
	SGD_niter = SGD_niter + niter;
end
println(TSGD_error/samplePaths);
println(TSGD_niter/samplePaths);
println(SGD_error/samplePaths);
println(SGD_niter/samplePaths);
=#

############################################################

##### Table 1 Noisey Rosenbrock ###############################
println("Noisy Rosenbrock Data");

############################################################

samplePaths = 2; 
TSGD_error, TSGD_niter = 0,0;
SGD_error, SGD_niter = 0,0; 
tol = 1e-1;
alpha = 1e-15;
x_0 = (500,500)
for i = 1:samplePaths
	#error,niter =  TSGD(x_0, alpha, tol, stochastic_grad, 1);
	#TSGD_error = TSGD_error + error; 
	#TSGD_niter = TSGD_niter + niter;
	error, niter = SGD(x_0, alpha, tol, stochastic_grad, 1);
	SGD_error = SGD_error + error; 
	SGD_niter = SGD_niter + niter;
end
#println(TSGD_error/samplePaths);
#println(TSGD_niter/samplePaths);
println(SGD_error/samplePaths);
println(SGD_niter/samplePaths);




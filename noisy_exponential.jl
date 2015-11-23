using Distributions
using Convex
using ECOS
using PyPlot

solver = ECOSSolver(verbose = false)
set_default_solver(solver);
d = Normal(0,1);
d = Uniform(-1,1);
d = Laplace(0,50)

function grad(x)
	return exp(x) - 1;
end

function stochastic_grad(x, numSamples)
	avg_grad = 0; 
	for z in 1:numSamples;
		noise = rand(d,1)[1];
		avg_grad = avg_grad + grad(x) + noise
	end
	return avg_grad/numSamples;
end


function project(x, i)
	c = 100;
	#projects x to |x-10| < i/c
	if(x > 10 && x-10 > i/c)
		return i/c + 10;
	elseif(x < 10 && 10-x < -i/c)
		return -i/c - 10;
	end
end


#=
#############################################################

##Figure 3 Noisy Exponential ##############################

println("Noisy exponential figure 3")

############################################################


curr_x = 500;
temp_x = 9;
##step size
alpha = 0.1; 
epsilon = 10.0^(-8);
test_1 = false;

samples = [1, 10,100,1000];

for numSamples in samples
	println(numSamples);
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
				curr_x = temp_x - alpha*SG/(1 + alpha*abs(SG));
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
	title(string("TSGD n=",numSamples));
	xlabel("k");
	ylabel("x_k");
	savefig(string("../figures/exponential_n", numSamples, "_alpha01.png"));
	clf();
end

=#

#############################################################

##Table 2/3 Noisy Exponential ##############################

println("Noisy exponential Table 2/3")

############################################################

#=

function TSGD(x,alpha,epsilon, numSamples)
	temp_x = x + 10; 
	numSteps = 1; 
	while(true)
		if(abs(x - temp_x) < epsilon || numSteps >= 10.0^(5))
			break;
		end
		temp_x = x;
		SG = stochastic_grad(x,numSamples);
		x = x - alpha*SG/(1 + alpha*abs(SG));
		numSteps = numSteps + 1; 
	end
	return [round(numSteps,0), round(x,3)];

end

function NSGD(x, alpha, epsilon, numSamples)
	temp_x = x + 10; 
	numSteps = 1; 
	while(true)
		if(abs(x - temp_x) < epsilon || numSteps >= 10.0^(5))
			break;
		end
		temp_x = x;
		SG = stochastic_grad(x,numSamples);
		#x = x - alpha*SG/abs(SG);
		#diminishing step size
		x = x - (alpha/numSteps^2)*SG/abs(SG);
		numSteps = numSteps + 1; 
	end
	return [round(numSteps,0), round(x,3)];
end

curr_x_array = [1, 10, 100];
alphas = [0.1,0.01, 0.001, 0.0001];
numSamples = 50; 
epsilon = 10.0^(-8);

##TO DO: test  n as a function of alpha 

#generate the table for latex
for curr_x in curr_x_array
	for alpha in alphas 
			(n_tsgd,x_tsgd) = TSGD(curr_x,alpha, epsilon, numSamples);
			(n_nsgd,x_nsgd) = NSGD(curr_x, alpha, epsilon, numSamples);
			println(string(curr_x," & ", alpha, " & ", "(", x_nsgd, ", ", n_nsgd, ") & (", x_tsgd, ", ", n_tsgd, ") \\\\ " ));
	end
end

=#


#############################################################

##Table 2/3 Noisy Exponential ##############################

println("Noisy exponential Figure 4")

############################################################
#=
function NGD(x_0, alpha, tol, grad_fun, numSamples)
	niter = 1;
	SG = grad_fun(x_0,numSamples);
	while(norm(SG) > tol && niter < 10^3)
		x_0 = x_0 - (alpha/niter^2)*SG/norm(SG);
		niter = niter + 1; 
		SG = grad_fun(x_0,numSamples);
		if(isinf(x_0)) break; end
		if(isnan(x_0)) break; end
	end
	x_0 = abs(x_0);
	if(x_0 > 10 || isnan(x_0) || isinf(x_0))
		x_0 = 10; 
	end
	return x_0;
end

function TSGD(x_0, alpha, tol, grad_fun, numSamples)
	SG = grad_fun(x_0,numSamples);
	while(norm(SG) > tol)
		x_0 = x_0 - alpha*SG/(alpha*norm(SG) + 1);
		SG = grad_fun(x_0,numSamples);
	end
	x_0 = abs(x_0);
	if(x_0 > 10)
		println(x_0)
	end
	return abs(x_0);
end


numSamples = 50; 

for tol in [10.0^(-3), 10.0^(-4), 10.0^(-5), 10.0^(-6)]
	a_vec = logspace(-5,-1,50);
	x0_vec = [1 ,10, 100];
	tic();
	for x_0 in x0_vec
		#take average over numAvgs iterations
		ss = size(a_vec,1);
		sgd_line = zeros(ss);
		tsgd_line = zeros(ss);
		numAvgs = 5; 
		for z = 1:numAvgs
			#println(string("current iterate: ", z));
			for k in 1:ss
				alpha = a_vec[k];
				tsgd_line[k] = tsgd_line[k] + TSGD(x_0, alpha, tol, stochastic_grad, numSamples);
				sgd_line[k] = sgd_line[k] + NGD(x_0, alpha, tol, stochastic_grad,numSamples);
			end
		end
		sgd_line = sgd_line./numAvgs;
		tsgd_line = tsgd_line./numAvgs;

		loglog(a_vec, tsgd_line, label = string("TSGD: \$x_0\$=",x_0));
		loglog(a_vec, sgd_line, label = string("NGD: \$x_0\$=",x_0), marker = "x");
	end
	toc();
	legend(fancybox="true")
	ax = gca() # Get the handle of the current axis
	ax[:set_ylim]([10.0^(-5), 10.0^(1.1)]);# Set the y axis to a logarithmic scale
	title(string("Noisy exponential tol=",tol));
	xlabel("\$\\alpha\$");
	ylabel("absolute error");
	tol = string(tol);
	tol = replace(tol, ".0", "");
	savefig(string("../figures/noisyexponential_tol",tol,".png"));
	clf();
end

=#
################################################
#### new convergence criteria TEST ####################
################################################

#=

x0=100;

alpha=0.0001;
sigma=1;
n=50;


avg_var = 0; 

tol = 0.2773;



X0 = x0; 
niter = 1; 


##for calculating running average of variance
n_var = 0; 

while(true)
	niter = niter+1;

	##noisy estimate of gradient 
	noise = sigma*randn(n,1);

	var_noise = var(noise);
	mean_noise = mean(noise);

	avg_var = avg_var + var_noise;
	n_var = n_var + 1;
	curr_var = avg_var/n_var;

	ngrad = grad(X0) + mean_noise;

	ttol = tol - sqrt(curr_var)/sqrt(n)*1.96;

#	println(string("ttol: ", ttol));
#	println(string("gradient: ",norm(ngrad)));
#	println(string("avg variance: ", curr_var));
	if(norm(ngrad) < ttol) 
		println(string("grad final: ", ngrad));
		break; 
	end;

	X1 = X0 - alpha*ngrad/(1 + alpha*norm(ngrad));
	#X1 = X0 - alpha*ngrad;

	#iterate
	X0 = X1;
end

println(string("Xfinal:",X0," niters:",niter));
println();

X0 = x0; 
niter = 1; 
while(true)
	niter = niter+1;

	ngrad = grad(X0)

	if(norm(ngrad) < tol) break; end;

	X1 = X0 - alpha*ngrad/(1 + alpha*norm(ngrad));

	#iterate
	X0 = X1;
end

println(string("grad_final", grad(X0)));
println(string("Xfinal:",X0," niters:",niter));

=#


### 
#testing small alpha



alphaVals = [1e-1,1e-2,1e-3,1e-4, 1e-5, 1e-6, 1e-7];

for alpha in alphaVals

	x0=100;

	##for calculating running average of variance
	n_var = 0; 

	path1 = Float64[];
	maxiter = 1e6;


	X1 = x0; 
	for niter = 1:maxiter
		ngrad = grad(X1) + randn(1);

		X1 = X1 - alpha*ngrad/(1 + alpha^2*norm(ngrad));
		X1 = X1[1];
		push!(path1,X1);
	end

	println(X1);

	ylabel("iterat value")
	xlabel("time")
	loglog((1:maxiter)*alpha,path1, label = "SA with step size alpha", color = "blue", marker = "<", linewidth = 0.5);



	X1 = x0; 
	path2 = Float64[];

	for niter = 1:maxiter
		ngrad = grad(X1);
		X1 = X1 - alpha*ngrad/(1 + alpha^2*norm(ngrad));
		X1 = X1[1];

		push!(path2,X1);
	end
	println(X1);

	loglog((1:maxiter)*alpha,path2, label = "ODE", color = "red", marker = "x", linewidth = 0.5);
	title(string("alpha = ",alpha));
	legend();
	titlename = string("../figures/noisy_exponential_ode_",alpha,".png");
	savefig(titlename);
	clf();
end


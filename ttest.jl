using PyPlot

function exp_grad(x)
	return exp(x) - 1;
end

function two_grad(x)
	return 2*x;
end

function four_grad(x)
	return 4*x^3;
end

function six_grad(x)
	return 6*x^5;
end


x0=0.5;
alpha=0.1;
n=50;
N = 50; 


#=
avgs = Float64[];
N_path = logspace(1,4,100);

for N in N_path
##for calc
avg_fin = 0; 

#
nniter = 1;

	numSamplePaths = 1e3;

	for n = 1:numSamplePaths
		X0 = x0; 
		#println(avg_fin/nniter);
		for i = 1:N 
			##noisy estimate of gradient 
			noise = 1*randn(1);

			#ngrad = exp_grad(X0) + mean(noise);
			#ngrad = two_grad(X0) + mean(noise);
			#ngrad = four_grad(X0) + mean(noise);
			ngrad = six_grad(X0) + mean(noise);


			#X1 = X0 - alpha*ngrad/(1 + alpha*norm(ngrad));

			X1 = X0 - (alpha/i)*ngrad;

			#iterate
			X0 = X1;
		end
		avg_fin = avg_fin + abs(X0);
		nniter = nniter + 1; 
	end
	println(string("N: ", N, " Average: ", avg_fin/nniter));
	push!(avgs,avg_fin/nniter);
end

loglog(N_path, avgs);
ylabel("Average over 1e3 sample paths");
xlabel("number of iterations")
title("SA optimizing f(x)");
#savefig(string("../figures/SA_exp.png"));
#savefig(string("../figures/SA_2.png"));
#savefig(string("../figures/SA_4.png"));
savefig(string("../figures/SA_6.png"));

readline();

=#


### two dimensional case

using Distributions
d = Normal();



function grad(vector)
	x = vector[1];
	y = vector[2];
	return [x; x];
end

function stochastic_grad(vector)
	return grad(vector) + 0.1*rand(d,2);
end



x0 = [0.5;0.5];

avgs = Float64[];
N_path = logspace(1,4,100);
numSamplePaths = 1e3;


for N in N_path
	##for calc
	avg_fin = 0; 
	nniter = 0;
	for z = 1:numSamplePaths
		X0 = x0;
		for i = 1:N
			X0 = X0 - (1/i)*stochastic_grad(X0);
		end
		avg_fin = avg_fin + norm(X0)^2;
		nniter = nniter + 1;
	end
	println(string("N: ", N, " Average Error: ", avg_fin/nniter));
	push!(avgs,avg_fin/nniter);
end
loglog(N_path, avgs);
ylabel("Average over 1e3 sample paths");
xlabel("number of iterations")
title("SA optimizing (1/2)x^Tx");
readline();
savefig(string("../figures/SA_22dim.png"));


######
## Run tests on the gradient in the asian option problem 
######
using Convex
using ECOS
using Distributions
using PyPlot

solver = ECOSSolver(verbose = false)
set_default_solver(solver);

function phi(x)
	##array stores price of assets at time jT/k
	S = zeros(size(x,1));
	for j in 1:size(x,1)
		S[j] = S_0*exp( (r-0.5*sigma^2)*(j*T/size(x,1)) + sigma*sqrt(T/size(x,1)) *sum(x[1:j]));
	end
	return exp(-r*T)*max(mean(S) - K, 0);
end

function phiw(w)

	S = zeros(size(w,1));
	for j in 1:size(w,1)
		S[j] = S_0 *exp((r - 0.5*sigma^2)*(j*T/size(w,1)) + sigma*w[j]);
	end
	return exp(-r*T)*max(mean(S) - K,0);


	#=
	val = 0;
	for i = 1:N
		val = val + S_0*exp( (r - 0.5*sigma^2)*(i*T/N) + sigma*w[i]);
	end
	val = val/N - K;
	return exp(-r*T)*max(val,0);
	=#
end




##the true distribution of X
function pdfTargetDist(x)
	mu = zeros(size(x,1));
	sigma = ones(size(x,1));
	targetDist = MvNormal(mu,sigma);
	return pdf(targetDist,x)
end


#what you're modeling the importance distribution as 
function pdfImportanceDist(x,theta)
	sigma = ones(size(theta,1));
	importDist = MvNormal(theta,sigma);
	return pdf(importDist,x)
end


##generate a sample from the multivariate-k standard
##normal distribution with mean shifted by theta. 
function generateSample(theta)
	sigma = ones(size(theta,1));
	importDist = MvNormal(theta,sigma);
	return rand(importDist);
end

function project(theta)
	theta = round(theta,5);
	theta_proj = Variable(size(theta,1));
	p = minimize(norm_2(theta_proj-theta))
	p.constraints += [abs(theta_proj) <=0.5];
	solve!(p)
	if(p.status == :Error)
		#println(string("Errortheta: ",theta));
	end
	return vec(theta_proj.value);
end


##returns the sample variance of the importance sampled estimator with sampling distribution f_\theta
function getSampleVariance(theta)
	numSamples = 10^6;
	##compute sample variance of estimator
	seq = Float64[];
	for i=1:numSamples
		x = generateSample(theta);
		val = phi(x)*exp(0.5*transpose(theta)*theta - transpose(x)*theta);
		push!(seq, val[1]);
	end
	return var(seq);
end


#parameters

#initial price of underlying asset
S_0 = 50;
K = 50;

#interest rate 
r = 0.05

#volatility 
sigma = 0.3;

T = 1.0

#parameter space size 
#used to be specified as k 
N = 50;

#constant in gradient desent
C = 0.01;




t_grid=linspace(0,T,N+1); # time grid
h=t_grid[2]-t_grid[1];     # time step size

t_grid=t_grid[2:end];

a1=r-0.5*sigma^2;
a2=sigma;
function phiN(x)
	return exp(-r*T)*max(mean(S_0*exp(a1*t_grid+a2*x))-K,0);
end

## calculate and save V' and V(theta) on a fine grid 
theta_grid = linspace(-10, 10,int(1e3));
meanArray = Float64[];
varArray = Float64[];
vprimeArray = Float64[];

numSamples = 1e7;
#alpha = 1e-8;

for theta in theta_grid
	println(theta);
	#gradArray = Float64[];
	guessArray = Float64[];
	guessArray2 = Float64[];
	for i = 1:numSamples
		x  = randn(N);
	    w=theta*t_grid+sqrt(h)*cumsum(randn(N,1));
		#likeratio = exp(-w[N]*theta + 0.5*theta^2*T);
		#curr_guess = phiN(w)*likeratio;
		#push!(guessArray, curr_guess);
		#push!(gradArray, (-w[end]+T*theta)*phiN(w)^2*exp(-2.0*w[end]*theta+T*theta^2));
		grad = (-w[end]+T*theta)*phiN(w)^2*exp(-2.0*w[end]*theta+T*theta^2);
		#dis = ((alpha/(1 + alpha*norm(grad)))*grad);
		#dis = alpha*grad;
		#push!(guessArray, dis);
		push!(guessArray2, grad);
	end
	#push!(varArray, var(guessArray));
	#push!(meanArray, mean(guessArray));
	push!(vprimeArray, mean(guessArray2));
end
#writedlm("data/asian_varFINE.txt", varArray);
#writedlm("data/asian_meanFINE.txt",meanArray);
writedlm("data/asianvprimeLARGE.txt", vprimeArray);
#writedlm("data/asian_mean.txt", meanArray);
#writedlm("data/asian_var.txt", varArray);
writedlm("data/theta_gridLARGE.txt",theta_grid);

##thse two data files will store the mean and variance of the displacements
## when alpha = 1e-4;
## at theta for 1e6 samples for a fine mesh of theta




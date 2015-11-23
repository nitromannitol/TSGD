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
S_0 = 100;
K = 100;

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


alpha = 1e-5;
numSteps = 1e4;

#theta is a constant 
theta = 3; 

t_grid=linspace(0,T,N+1); # time grid
h=t_grid[2]-t_grid[1];     # time step size

t_grid=t_grid[2:end];

a1=r-0.5*sigma^2;
a2=sigma;
function phiN(x)
	return exp(-r*T)*max(mean(S_0*exp(a1*t_grid+a2*x))-K,0);
end



nSteps = 1e7; 
alphaArray = logspace(0,-6,100);
autoCorArray = Float64[];


tic()
for alpha in alphaArray
	println(alpha);
	guessArray = Float64[];
	x_0 = 1;
	for i = 1:nSteps
		h = T/N;
		w=theta*t_grid+sqrt(h)*cumsum(randn(N,1));
		gradient=(-w[end]+T*theta)*phiN(w)^2*exp(-2.0*w[end]*theta+T*theta^2);
		theta = theta - (alpha/(alpha*norm(gradient) + 1))*gradient;
		push!(guessArray,theta[1]);
	end
	fileName = string("tempfile",int(rand(1)*1e5));
	writedlm(fileName, guessArray);
	a = readall(`acorTarball/acor $fileName`);
	rm(fileName);
	println(a);
 	mm = match(r"autocorrelation time = \d++(?>\.\d++)?(?>e[+-]?\d++)?", a);
 	substring = mm.match;
 	println(substring);
 	auto_cor_time = mm.match[length("autocorrelation time = "):end];
 	println(auto_cor_time);
 	auto_cor_time = float(auto_cor_time);
 	push!(autoCorArray,auto_cor_time);
end


loglog(alphaArray,autoCorArray);
title(string("TSGD on asian option,#steps = ", nSteps, ", x_0 = ", 1));
xlabel("alpha");
ylabel("integrated autocorrelation time");
savefig("asian_autocortime.png");
println(alphaArray);


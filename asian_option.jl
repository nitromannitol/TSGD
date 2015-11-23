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
		S[j] = S_0*exp( (r-0.5*sigma^2)j*T/size(x,1) + sigma*sqrt(T/size(x,1)) *sum(x[1:j]));
	end
	return exp(-r*T)*max(sum(S)/N - K, 0);
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
	p.constraints += [abs(theta_proj) <=1];
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
N = 200;

#constant in gradient desent
C = 0.01;



theta = ones(N);
alpha = 1e-5;

numSteps = 1e5;

guessArray = Float64[];


theta = linspace(0.25, 0, N);

for i = 1:numSteps
	x  = randn(N) + theta;
	push!(guessArray, (phi(x)*exp(0.5*transpose(theta)*theta - transpose(x)*theta))[1]);
end
TSGD_theta_opt = theta;
TSGD_var = getSampleVariance(theta);



thetaArray = zeros(int(numSteps),N);


for i = 1:numSteps
	alpha = 1/i;
	thetaArray[i,:] = theta;
	#generate a sample from f_theta
	#x = generateSample(theta);
	x  = randn(N) + theta;
	#x = randn(N);

	#g1 = (theta - x)*phi(x)^2*pdfTargetDist(x)^2/pdfImportanceDist(x,theta)^2;
	gradient = (theta - x)*phi(x)^2.*(exp(0.5*transpose(theta)*theta - transpose(x)*theta)).^2;

	#gradient = 2*(theta -x)*phi(x)^2.*exp(0.5*transpose(theta)*theta - transpose(x)*theta);


	theta = theta  - (alpha/(alpha*norm(gradient) + 1))*gradient;
	push!(guessArray, (phi(x)*exp(0.5*transpose(theta)*theta - transpose(x)*theta))[1]);
end
TSGD_theta_opt = theta;
println(theta);
println(string("TSGD variance during run: "), var(guessArray));
println(string("TSGD mean estimate: ", mean(guessArray)));
#=
writedlm("TSG_run.txt", guessArray);
for i = 1:N
	writedlm(string("acorTarball/TSG_run_",i,"_.txt"),thetaArray[:,i]);
end

TSGD_var = getSampleVariance(theta);
=#



###compare to ADAMC
theta = ones(N);

for i = 1:numSteps
	x = generateSample(theta);
	gradient = (theta - x)*phi(x)^2.*(exp(0.5*transpose(theta)*theta - transpose(x)*theta)).^2;
	theta = theta-C/sqrt(i)*gradient;
	if(maximum(abs(theta)) > 0.5)
		theta = project(theta);
	end
end
ADAMC_theta_opt = theta;
ADAMC_var = getSampleVariance(theta);

println(string("Optimal ADAMC theta: ", ADAMC_theta_opt));
println(string("Variance: "),ADAMC_var);
println(string("Optimal TSGD theta: ", TSGD_theta_opt));
println(string("Variance: "), TSGD_var);




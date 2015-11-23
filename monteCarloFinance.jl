using Convex
using ECOS
using Distributions
using PyPlot

solver = ECOSSolver(verbose = false)
set_default_solver(solver);

#parameters

#initial price of underlying asset
S_0 = 50;
K = 50;

#interest rate 
r = 0.05

#volatility 
sigma = 0.3;

T = 1.0

#number of time grid points
N = 64;

#constant in gradient desent
C = 0.01;





function phi(x)
	##array stores price of assets at time jT/N
	S = zeros(N)
	for j in 1:N
		S[j] = S_0*exp( (r-0.5*sigma^2)j*T/N + sigma*sqrt(T/N) *sum(x[1:j]));
	end
	return exp(-r*T)*max(sum(S)/N - K, 0);
end


##the true distribution of X
function pdfTargetDist(x)
	mu = zeros(N);
	sigma = ones(N);
	targetDist = MvNormal(mu,sigma);
	return pdf(targetDist,x)
end


#what you're modeling the importance distribution as 
function pdfImportanceDist(x,theta)
	sigma = ones(N);
	importDist = MvNormal(theta,sigma);
	return pdf(importDist,x)
end


##generate a sample from the multivariate-k standard
##normal distribution with mean shifted by theta. 
function generateSample(theta)
	sigma = ones(N);
	importDist = MvNormal(theta,sigma);
	return rand(importDist);
end

function project(theta)
	theta = round(theta,3);
	theta_proj = Variable(N);
	p = minimize(norm_2(theta_proj-theta))
	p.constraints += [abs(theta_proj) <=0.5];
	solve!(p)
	if(p.status == :Error)
		#println(string("Errortheta: ",theta));
	end
	return theta_proj.value
end


true_price = 4.02; 
numStepArrayLen = 30; 
numStepArray = logspace(3,5,numStepArrayLen);
numStepArray = [10^6];
erArray = Float64[];
stdArray = Float64[];
for numSteps in numStepArray
	tic();
	currGuessArray = Float64[];
	for i in 1:numSteps
		x = randn(N);
		push!(currGuessArray, phi(x)); 
	end
	toc();
meanVal = mean(currGuessArray);
push!(stdArray, sqrt(var(currGuessArray)));
println(mean(currGuessArray));
push!(erArray, abs(meanVal - true_price)/abs(true_price));
end
errorbar(numStepArray, erArray, stdArray.*1.96./numStepArrayLen);
xlabel("number of steps");
ylabel("relative error");
title("monte carlo asian option");
savefig("../../figures/asian_option_mc.png");
println("Done:");
readline();



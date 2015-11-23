######
## Test the autocorrelation time of running truncated gradient descent on asian option
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


alpha = 1e-5;
nbins = 100;

tic();
println(alpha);
guessArray = Float64[];
theta = 0;
nSteps = 1e7;
for i = 1:nSteps
	h = T/N;
	w=theta*t_grid+sqrt(h)*cumsum(randn(N,1));
	gradient=(-w[end]+T*theta)*phiN(w)^2*exp(-2.0*w[end]*theta+T*theta^2);
	theta = theta - (alpha/(alpha*norm(gradient) + 5))*gradient;
	theta = theta - alpha*gradient;
	push!(guessArray,theta);
end
println(string("current mean theta: " , mean(guessArray)));


##generate PDF 
zz = hist(guessArray,nbins)
pdf = zz[2]./sum(zz[2])
thetaBucketsMean = Float64[];
bucketsAr = zz[1];
for i = 1:(length(bucketsAr)-1)
	mVal = (bucketsAr[i] + bucketsAr[i+1])/2
	push!(thetaBucketsMean, mVal);
end

thetaArray = thetaBucketsMean;



#compute V(theta) over same range of theta 
#thetaArray = linspace(0,2.5,30);
vArray = Float64[];
numSamples = 1e6;

for theta in thetaArray
	zz = Float64[];
	for i=1:numSamples
		w=theta*t_grid+sqrt(h)*cumsum(randn(N,1));
		#w = sqrt(h)*cumsum(randn(N,1));
		likeratio = exp(-w[N]*theta + 0.5*theta^2*T);
		c = phiN(w)*likeratio;
		#c = phiN(w)^2*likeratio;
		push!(zz, c);
	end
	println(var(zz));
	push!(vArray, var(zz));
end
expV = exp(-vArray)./sum(exp(-vArray));
plot(thetaArray, -log(expV), label = "V(theta)");
plot(thetaArray, -log(pdf), label = "SGD");
legend();
ax = gca();
ax[:set_yscale]("log");
title("1D Asian Option on SGD with alpha=1e-5");
xlabel("theta");
ylabel("-log(PDF)");
savefig("SGD_ASIA_PDF.png")

println(thetaArray);












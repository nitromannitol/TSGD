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

theta_0 = 4; 

endtime = 1; 

thetastar = 0.9379999;

alphaVals = [1e-1,1e-2,1e-3,1e-4,1e-5, 1e-6, 1e-7, 1e-8,1e-9,1e-10];

for alpha in alphaVals
	theta = theta_0;

	path1 = Float64[];
	nsteps = endtime/alpha;
	#nsteps = 1e6;

	##TSGD SOLUTION

	for i = 1:nsteps
		h = T/N;
		w=theta*t_grid+sqrt(h)*cumsum(randn(N,1));
		grad = (-w[end]+T*theta)*phiN(w)^2*exp(-2.0*w[end]*theta+T*theta^2);
		theta = theta - (alpha/(1 + alpha*norm(grad)))*grad;
		push!(path1, theta);
	end
	#println(theta);
	println(string("Done with TSGD, starting SDE for alpha = "),alpha);
	plot(1:nsteps, path1, label = "TSGD");
	#ax = gca();
	#ax[:set_xscale]("log");
	theta = theta_0;
	path2 = Float64[];

	true_gradAr = readdlm("data/asianvprimeLARGE.txt");



	## tamed SDE SOLUTION
	## TO DO: IT"S HARDCODED BELOW, WILL BREAK IF THETA GRID CHANGES
	for i = 1:nsteps
		h = T/N;
		w=theta*t_grid+sqrt(h)*cumsum(randn(N,1));
		grad = (-w[end]+T*theta)*phiN(w)^2*exp(-2.0*w[end]*theta+T*theta^2);

		#get true gradient from array
		#println(theta);
		ind = int( 1e3*(theta+10)/(10+10));
		
		if(ind > 0 && ind < 1e3)
			true_grad = true_gradAr[ind];
		#	println(string("true gradient: ", true_grad, "gradient: ", grad));
			theta = theta - (alpha/(1 + alpha*norm(true_grad)))*grad;
		else
			theta = theta - (alpha/(1 + alpha*norm(grad)))*grad;
		end
		push!(path2, theta);
	end

	plot(1:nsteps, path2, label = "SDE");
	#loglog((1:nsteps)*alpha,path2, label = "ODE", color = "red", marker = "x", linewidth = 0.5);

	legend(loc = "lower left");
	title(string("SGD VS SDE on 1D Asian with alpha = ",alpha));
	titlename = string("SDE_COMPARISON",alpha,".png");
	xlabel("time");
	ylabel("theta");
	savefig(titlename);
	clf();
	println(string("Done with SDE for alpha = "),alpha);
	#println(string("final theta: ", theta));
end


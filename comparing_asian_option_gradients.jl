using Distributions


function phi(x)
	##array stores price of assets at time jT/k
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
	theta = round(theta,5);
	theta_proj = Variable(N);
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
N = 64;

#constant in gradient desent
C = 0.01;



#noisy gradient when sampling from normal distriubiton
grad1_ar = Float64[];

#noisy gradient when sampling from importance distribution
grad2_ar = Float64[];

numSteps = 1e7;

N = 1; 

#fix theta
theta = 5*ones(N);


for i in 1:numSteps
	x_1 = randn(N);
	grad1 = (theta - x_1)*phi(x_1)^2.*(exp(0.5*transpose(theta)*theta - transpose(x_1)*theta));

	x_2 = randn(N) + theta;
	grad2 = (theta - x_2)*phi(x_2)^2.*(exp(0.5*transpose(theta)*theta - transpose(x_2)*theta)).^2;

	push!(grad1_ar, grad1[1]);
	push!(grad2_ar,grad2[1]);
end

mean_G1 =  mean(grad1_ar);
mean_G2 = mean(grad2_ar);

println(string("Mean when sampling from standard normal ", mean_G1));
println(string("Mean when sampling from standard normal shifted by theta ", mean_G2));
println(norm(mean_G1-mean_G2)/norm(mean_G1));





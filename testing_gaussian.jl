

numSteps = 10^5; 

N = 64; 
theta = 5000*ones(N);

thetaArray = zeros(int(numSteps),N);

alpha = 1e-2;

for i = 1:numSteps
	thetaArray[i,:] = theta;
	#theta = theta - (alpha)/(alpha*norm(theta) + 1)*theta;
	theta = theta - alpha*theta;
end

println(theta);

for i = 1:N
	writedlm(string("acorTarball/gaussian_run/TSG_run_",i,"_.txt"),thetaArray[:,i]);
end
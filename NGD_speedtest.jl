using Distributions
using Convex
using ECOS
using PyPlot


d = Normal(0,10);

function grad(x)
	return 2x; 
end

function stochastic_grad(x, numSamples)
	avg_grad = 0; 
	for z in 1:numSamples;
		noise = rand(d,1)[1];
		avg_grad = avg_grad + grad(x) + noise
	end
	return avg_grad/numSamples;
end



curr_x = 500;
temp_x = 9;
##step size
alpha = 0.1; 
epsilon = 0.001;
test_1 = false;


times = Float64[];


numSampleRuns = 5000; 
avgSteps = 0; 


for i in 1:numSampleRuns
	numSteps = 0; 
	curr_x = 500;
	numSamples = 5; 
	for z in 1:100000
		SG = stochastic_grad(curr_x,numSamples);
		#curr_x = curr_x - alpha*SG/(alpha*norm(SG)+1); 
		curr_x = curr_x - alpha*SG/norm(SG);
		numSteps = numSteps+1;
		if(norm(curr_x) < epsilon)
			break;
		end
	end
	avgSteps = avgSteps + numSteps
end
println(string("final x: ", curr_x));
println(string("average steps to convergence, ", avgSteps/numSampleRuns));
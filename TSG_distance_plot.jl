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



times = Float64[];


numSteps = 10000;
alphas = [0.001,0.01,0.1];


for alpha in alphas
	curr_x = 500;
	sampleArray = [1,50,1000];
	for numSamples in sampleArray
		stepsizes = Float64[];
		for i in 1:numSteps
			SG = stochastic_grad(curr_x,numSamples);
			curr_x = curr_x - alpha*SG/(alpha*norm(SG)+1); 
			push!(stepsizes,(alpha*SG/(alpha*norm(SG)+1))^2); 
		end
		semilogy(1:numSteps,stepsizes);
		alpha = round(alpha,5);
		title(string("TSGD n=",numSamples, " alpha =",alpha));
		xlabel("numSteps");
		ylabel("stepSize");
		ax = gca() # Get the handle of the current axis
		ax[:set_ylim]([10.0^(-15),1]);# Set the y axis to a fixed scale
		astring = string(alpha);
		astring = replace(astring, ".","p");
		savefig(string("paper/figures/TSG_SD/alpha=", astring, "n=",numSamples,".png"));
		#print("Hit <enter> to continue")
		#readline()
		clf();
	end

end
using Distributions
using Convex
using ECOS
using PyPlot

solver = ECOSSolver(verbose = false)
set_default_solver(solver);
d = Normal(0,10);

function grad(x)
	return 2*x; 
end

function stochastic_grad(x, numSamples)
	avg_grad = 0; 
	for z in 1:numSamples;
		noise = rand(d,1)[1];
		avg_grad = avg_grad + grad(x) + noise
	end
	return avg_grad/numSamples;
end


curr_x = 10;
temp_x = 9;
##step size
alpha = 0.1; 
epsilon = 0.001;
test_1 = true;

samples = logspace(2,6,50)
opt_sols = Float64[];


for numSamples in samples
	#println(numSamples);
	numSampleRuns = 1; 
	numSteps = 10000;
	avg = zeros(numSteps);
	for i = 1:numSampleRuns
		curr_x = 500;
		current_xs = Float64[];
		for i = 1:numSteps
			temp_x = curr_x;

			if(test_1)
				curr_x = temp_x - alpha*stochastic_grad(temp_x,numSamples);
				if(curr_x > i)
					#curr_x = project(curr_x,i);
				end
			else
				SG = stochastic_grad(temp_x,numSamples);
				curr_x = temp_x - alpha*SG/(1 + alpha*abs(SG));
			end
			#println(i);
			push!(current_xs,curr_x);
		end
		#plot(1:numSteps, current_xs, linestyle = "--", linewidth = 0.5);
		avg = avg + current_xs;
	end
	avg_opt = (avg/numSampleRuns)[numSteps];
	println(string("Num samples: ", numSamples, " Optimal: ", avg_opt));
	push!(opt_sols, abs(avg_opt));
	#plot(1:numSteps, avg/numSampleRuns, linewidth = 3, color = "black");
	#ax = gca() # Get the handle of the current axis
	#ax[:set_ylim]([-0.2,0.2]);# Set the y axis to a logarithmic scale
	#title(string("TSGD n=",numSamples));
	
	#savefig(string("exponential_n", numSamples, "_alpha01.png"));
	#readline()
	#clf();
end
loglog(samples, opt_sols)
loglog(samples,1./sqrt(samples));
xlabel("n");
ylabel("x*");
print("Hit <enter> to continue")
readline();


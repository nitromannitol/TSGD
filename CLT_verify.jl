using Distributions
using Convex
using ECOS
using PyPlot

samples = logspace(2,6,100)
opt_sols = Float64[];


d = Normal(0,10);


for numSamples in samples
	avg = 0; 
	numSamples = round(numSamples);
	for i = 1:numSamples
		avg = avg + rand(d,1)[1];
	end
	avg = abs(avg/numSamples);
	push!(opt_sols, avg)
end

loglog(samples,opt_sols);
loglog(samples,1./sqrt(samples));
xlabel("n");
ylabel("x*");
println("Hit <enter> to continue")
readline();

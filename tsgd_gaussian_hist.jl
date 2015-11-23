######
## Find the stationary distribution of TSGD when minimizing the function f(x) = x^2/2
######
using Convex
using ECOS
using Distributions
using PyPlot

solver = ECOSSolver(verbose = false)
set_default_solver(solver);


alpha = 1e-1;
nbins = 50;

tic();
guessArray = Float64[];
nSteps = 1e7;
x_0 = 10; 
for i = 1:nSteps
	gradx = x_0 + randn(1);
	x_0 = x_0 - (alpha/(1 + norm(gradx)*alpha))*gradx;
	push!(guessArray,x_0[1]);
end
println(string("current mean x: " , mean(guessArray)));
#println(typeof(guessArray));
#println(guessArray);
ww = ones(int(nSteps))./nSteps;
#histObject = hist(guessArray, nbins, weights = ww);
plt[:hist](guessArray, nbins, weights = ww);
xlabel("x");
ax = gca();
#ax[:set_yscale]("log");
title(string("TSGD on gaussian with alpha = ", alpha));
savefig(string("TSGD_GAUSSIAN_HISTOGRAM_alpha",alpha,".png"));
clf();











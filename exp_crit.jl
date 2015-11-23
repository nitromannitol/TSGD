######
## Test the accuracy of TSGD on f(x) = exp(x) - x 
######
using PyPlot

#f(x) = cosh(x)

nSteps = 1e6; 


alphaArray = logspace(-5,0,50);
meanArray = Float64[];
stdArray = Float64[];

tic();
for alpha in alphaArray
	println(alpha);
	guessArray = Float64[];
	x_0 = 10;
	for i = 1:nSteps
		gradx = exp(x_0) - 1 + randn(1);
		#x_0 = x_0 - (alpha/(alpha*norm(gradx) + 1))*gradx;
		x_0 = x_0 - (alpha/(alpha*norm(gradx) + 1))*gradx;
		#x_0 = x_0 - (alpha/norm(gradx))*gradx
		push!(guessArray,x_0[1]);
	end
	println(string("current mean: " , mean(guessArray)));
	push!(stdArray,sqrt(var(guessArray)));
	push!(meanArray, abs(mean(guessArray)));
end
toc();
errorbar(alphaArray,meanArray,stdArray);
ax = gca();
ax[:set_yscale]("log");
ax[:set_xscale]("log");
xlabel("alpha");
ylabel("relative error after 1e6 steps");
title("TSGD on noisy exponential with x_0 = 10")
savefig("nsgd_exp_critalpha.png");
println("done");
readline();
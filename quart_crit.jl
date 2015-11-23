######
## Test the autocorrelation time of running truncated gradient descent on f(x) = 1/2 x^T x with gradient grad f(x) = x + randn
######
using PyPlot


nSteps = 1e6; 


alphaArray = logspace(-10,4,100);
meanArray = Float64[];
stdArray = Float64[];

tic();
for alpha in alphaArray
	println(alpha);
	guessArray = Float64[];
	x_0 = 10;
	for i = 1:nSteps
		gradx = x_0.^3 + randn(1);
		x_0 = x_0 - (alpha/(alpha*norm(gradx) + 1))*gradx;
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
title("TSGD on noisy quartic with x_0 = 10")
savefig("quart_critalpha.png");
println("done");
readline();
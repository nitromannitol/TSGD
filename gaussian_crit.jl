######
## Test the autocorrelation time of running truncated gradient descent on f(x) = 1/2 x^T x with gradient grad f(x) = x + randn
######
using PyPlot


nSteps = 1e6; 


alphaArray = logspace(-4,0,100);
meanArray = Float64[];
stdArray = Float64[];

zetaMeanArray = Float64[];
zetaStdArray = Float64[];

nStepsAlpha = 1e3;

tic();
for alpha in alphaArray
	println(alpha);
	guessArray = Float64[];
	zetaArray = Float64[];
	x_0 = 10;
	nSteps = nStepsAlpha/alpha;
	for i = 1:nSteps
		gradx = x_0 + randn(1);
		x_0 = x_0 - (alpha/(alpha*norm(gradx) + 1))*gradx;
		push!(guessArray,x_0[1]);
		push!(zetaArray, (alpha/(alpha*norm(gradx) + 1)));
	end
	#println(string("current mean: " , mean(guessArray)));
	push!(stdArray,sqrt(var(guessArray)));
	push!(meanArray, abs(mean(guessArray)));
	push!(zetaMeanArray, mean(zetaArray));
	push!(zetaStdArray, sqrt(var(zetaArray)));
end
toc();
errorbar(alphaArray,meanArray,stdArray);
ax = gca();
ax[:set_yscale]("log");
ax[:set_xscale]("log");
xlabel("alpha");
ylabel("relative error after nstepsoveralpha steps");
title("TSGD on noisy gaussian with x_0 = 10")
savefig("gaussian_critalpha.png");
clf();
errorbar(alphaArray,zetaMeanArray,zetaStdArray);
ax = gca();
ax[:set_yscale]("log");
ax[:set_xscale]("log");
xlabel("alpha");
ylabel("mean zeta after nstepsoveralpha steps");
title("zeta for TSGD on noisy gaussian")
savefig("gaussian_zeta.png");
println("done");
readline();

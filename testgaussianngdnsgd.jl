######
## Test the autocorrelation time of running truncated gradient descent on f(x) = 1/2 x^T x with gradient grad f(x) = x + randn
######
using PyPlot


nSteps = 1e6; 

alphaArray = logspace(-10,10,100);
nsgdMeanArray = Float64[];
stdArray = Float64[];

zetaMeanArray = Float64[];
zetaStdArray = Float64[];


nsteps = 1e6; 

tic();
for alpha in alphaArray
	println(alpha);
	guessArray = Float64[];
	zetaArray = Float64[];
	x_0 = 10;
	for i = 1:nSteps
		gradx = x_0 + randn(1);
		x_0 = x_0 - (alpha/(alpha*norm(gradx) + 1))*gradx;
		push!(zetaArray, (alpha/(alpha*norm(gradx) + 1)));
		push!(nArray, alpha)
	end
	push!(stdArray,sqrt(var(guessArray)));
	push!(meanArray, abs(mean(guessArray)));
	push!(zetaMeanArray, mean(zetaArray));
	push!(zetaStdArray, sqrt(var(zetaArray)));
end
toc();


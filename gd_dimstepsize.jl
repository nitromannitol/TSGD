######
## GD with diminishing step size and TGD
######
using PyPlot

#f(x) = cosh(x)


nSteps = 1e6; 

meanArray = Float64[];
stdArray = Float64[];


tic();
guessArray = Float64[];
x_0 = 10;
for i = 1:nSteps
	alpha = 1/i; 
	#gradx = exp(x_0) - 1;
	#gradx = sinh(x_0) + 1e4*randn(1)[1];
	gradx = sinh(x_0);
	#x_0 = x_0 - alpha*gradx;
	stepsize = alpha; 
	if(norm(gradx) > 1)
		stepsize = 1/norm(gradx);
	end
	#x_0 = x_0 - alpha/max(1,norm(gradx)*alpha)*gradx;
	#x_0 = x_0 - alpha/norm(gradx)*gradx;
	x_0 = x_0 - stepsize*gradx;
	#x_0 = x_0 - alpha*gradx
	push!(guessArray,x_0[1]);
end
push!(stdArray,sqrt(var(guessArray)));
push!(meanArray, abs(mean(guessArray)));
toc();
println(string("relative error final point: ",abs(x_0), " relative error mean: ", abs(mean(guessArray)), " variance: ", sqrt(var(guessArray))));
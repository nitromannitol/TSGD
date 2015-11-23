######
## When does GD fail? 
######
using PyPlot

#f(x) = cosh(x)


nSteps = 1e6; 


alphaArray = logspace(-20,0,50);
meanArray = Float64[];
stdArray = Float64[];
minVal = 999; 
minalpha = 0; 

tic();
for alpha in alphaArray
	guessArray = Float64[];
	x_0 = 30;
	prev_x = 31;
	for i = 1:nSteps
		#gradx = exp(x_0) - 1;
		gradx = sinh(x_0);
		#x_0 = x_0 - alpha*gradx;
		stepsize = alpha; 
		if(norm(gradx) < 1)
			stepsize = 1/norm(gradx);
		end
		x_0 = x_0 - stepsize*gradx;
		#x_0 = x_0 - (alpha/(alpha*norm(gradx) + 1))*gradx;
		#x_0 = x_0 - (alpha/(alpha*norm(gradx) + 1))*gradx;
		#x_0 = x_0 - (alpha/norm(gradx))*graadx
		push!(guessArray,x_0[1]);
	end
	println(string("current mean: " , abs(mean(guessArray))));
	if(abs(mean(guessArray)) < minVal)
		minVal = abs(mean(guessArray));
		println(minVal);
		minalpha = alpha;
	end
	push!(stdArray,sqrt(var(guessArray)));
	push!(meanArray, abs(mean(guessArray)));
end
toc();
println(string("minimum: ", minVal, "minalpha: ", minalpha));
errorbar(alphaArray,meanArray,stdArray);
ax = gca();
ax[:set_yscale]("log");
ax[:set_xscale]("log");
xlabel("alpha");
ylabel("relative error after 1e6 steps");
title("GD on exponential with x_0 = 10")
savefig("gd_exp_critalpha.png");
println("done");
readline();
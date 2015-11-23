######
## Test the autocorrelation time of running truncated gradient descent on f(x) = 1/2 x^T x with gradient grad f(x) = x + randn
######
using PyPlot


#dimension 
N = 1; 



nSteps = 1e7; 


alphaArray = logspace(0,-5,50);
autoCorArray = Float64[];


tic();
for alpha in alphaArray
	println(alpha);
	guessArray = Float64[];
	x_0 = 1;
	for i = 1:nSteps
		gradx = x_0 + randn(1);
		x_0 = x_0 - (alpha/(alpha*norm(gradx) + 1))*gradx;
		push!(guessArray,x_0[1]);
	end
	fileName = string("tempfile",int(rand(1)*1e5));
	writedlm(fileName, guessArray);
	a = readall(`acorTarball/acor $fileName`);
	rm(fileName);
	println(a);
 	mm = match(r"autocorrelation time = \d++(?>\.\d++)?(?>e[+-]?\d++)?", a);
 	substring = mm.match;
 	println(substring);
 	auto_cor_time = mm.match[length("autocorrelation time = "):end];
 	println(auto_cor_time);
 	auto_cor_time = float(auto_cor_time);
 	push!(autoCorArray,auto_cor_time);
end
toc();

loglog(alphaArray,autoCorArray, label = "experimental integrated auto cor time");
loglog(alphaArray, (2 .- alphaArray)./alphaArray, label = "theoretical integrated auto cor time");
legend();
title(string("TSGD on Gaussian,#steps = ", nSteps, ", x_0 = ", 1));
xlabel("alpha");
ylabel("integrated autocorrelation time");
savefig("gaussian_autocortime.png");
println(alphaArray);
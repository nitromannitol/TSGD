##plot discretization term 
#using PyPlot


#gradient 
function gradient(x)
	return exp(x) - 1;
end


numSteps = 1e6; 

x = 100; 
array = Float64[];

for i = 1:numSteps
	alpha = 1/(i*100);
	gradx = gradient(x);
	#stepsize = alpha/(max(1,alpha*norm(gradx)));
	stepsize = alpha
	x = x - stepsize.*gradx;
	discError = stepsize^2*norm(gradx)^2;
	println(discError);
	push!(array, discError[1]);
	println(x)
end
println(x)
#plot(1:numSteps, array);
#xlabel("numSteps");
#ylabel("discretization error");
#title(string("discretization error for f(x) = e^{e(x)}}"));
#ax = gca();
#ax[:set_yscale]("log");
#ax[:set_xscale]("log");
#savefig("discError.png");
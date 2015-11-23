using PyPlot

###
###
###
###

###comparing euler maruyama method solution to SDE 


##########
#
# Y_1 = Y_0 - dt \nabla u(Y_0) + sqrt(dt) sqrt(h) N(0,1)
#
##########

#### Y_{\floor{t/dt}} = X_{\floor{t/h}}

##########
#
# X_1 = X_0 - h ( \nabla u(x_0) + N(0,1))
#
##########



##consider f(x) = (1/4) x^4 
## then the updates become X_1 = X_0 - h( X_0^3 + N(0,1)) for SA
## and Y_1 = Y_0 - dt Y_0^3 + sqrt(dt) sqrt(h) N(0,1)



#fix h = 0.1



xVals = Float64[];
yVals = Float64[];


dt = 1e-4;

mean_diff = Float64[];
var_diff = Float64[];

hVals = logspace(-3,-0.8,50);

numIterates = 1e7; 

tic();
for h in hVals
	println(h);

	sde_time = 1:(1/dt);
	sa_time = 1:(1/h);

	x0 = 1;
	y0 = 1; 


	for i = 1:numIterates
		Y_0 = y0;
		#compare mean of SA after 1/h steps
		for t in sde_time
			Y_0 = Y_0 - dt*(Y_0^5) + sqrt(dt)*sqrt(h)*randn(1);
			Y_0 = Y_0[1];
		end
		push!(yVals, Y_0);
	end
	sde_mean = mean(yVals);
	sde_var = var(yVals);
	for i = 1:numIterates
		X_0 = x0;
		for t in sa_time
			X_0 = X_0 - h*(X_0^5 + randn(1));
			X_0 = X_0[1];
		end
		push!(xVals, X_0);
	end

	sa_mean = mean(xVals);
	sa_var = var(xVals);


	output = string("SA MEAN: ", sa_mean, " SA VAR:", sa_var, " SDE MEAN: ", sde_mean, " SDE VAR: ", sde_var);

	println(output)


	push!(mean_diff, abs(sa_mean - sde_mean)/abs(sde_mean));
	push!(var_diff, abs(sde_var - sa_var)/abs(sde_var));

end
toc();

ylabel("relative error");
xlabel("h");
title(string("dt =", dt, " numsamples = ", numIterates));
loglog(hVals, mean_diff, label = "relative error of means");
loglog(hVals, var_diff, label = "relative error of variance");
loglog(hVals, hVals, label = "f(x) = h");
legend();
titlename = string("../figures/SDE_TEST.png");
savefig(titlename);
println("Enter");
readline();

#=
ylabel("iterate value");
xlabel("time");
title("SA vs. SDE");
#loglog(sde_time,  yVals, label = "SDE path");
#loglog(sa_time, xVals, label = " SA path");
loglog(sde_time, abs(yVals - xVals), label = "absolute difference between SDE and SA");
legend();
println("Press enter.");
readline();
=#





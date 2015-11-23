###
using PyPlot
theta_grid = linspace(-10,10,int(1e3));


for i in [4,5,6,7,8,9]
	fileName = string("data/asian_var",i,".txt");
	var = readdlm(fileName);
	labelName = string("alpha=1e-",i);
	val = (10)^(2*i);
	plot(theta_grid, var.*val, label = labelName);
	#plot(theta_grid, -mean.*val, label = labelName);
	#titString = string("TSGD Asian alpha =1e-",i);
	#title(titString);
	xlabel("theta");
	ylabel("variance");
	figName = string("data/TSGDAsian_alpha_1e-",i,".png");
end
#vprime = readdlm("data/asianvprime.txt");
#plot(theta_grid, -vprime, label = "-V'");
ax = gca();
ax[:set_yscale]("log");
ax[:set_ylim]((1e1,1e7));
legend(loc = "upper left", labelspacing = 0.1);
title("TSGD on Asian Option");
savefig("data/diffusionGraphsVar.png");
using Convex
using ECOS
using Distributions
using PyPlot

solver = ECOSSolver(verbose = false)
set_default_solver(solver);

#parameters

#initial price of underlying asset
S_0 = 50;
K = 50;

#interest rate 
#r = 0.05
r = 0.1;

#volatility 
#sigma = 0.3;
sigma = 0.3;

T = 1.0

#parameter space size 
k = 64;

#constant in gradient desent
C = 0.01;





function phi(x)
	##array stores price of assets at time jT/k
	S = zeros(k)
	for j in 1:k
		S[j] = S_0*exp( (r-0.5*sigma^2)j*T/k + sigma*sqrt(T/k) *sum(x[1:j]));
	end
	return exp(-r*T)*max(sum(S)/k - K, 0);
end


##the true distribution of X
function pdfTargetDist(x)
	mu = zeros(k);
	sigma = ones(k);
	targetDist = MvNormal(mu,sigma);
	return pdf(targetDist,x)
end


#what you're modeling the importance distribution as 
function pdfImportanceDist(x,theta)
	sigma = ones(k);
	importDist = MvNormal(theta,sigma);
	return pdf(importDist,x)
end


##generate a sample from the multivariate-k standard
##normal distribution with mean shifted by theta. 
function generateSample(theta)
	sigma = ones(k);
	importDist = MvNormal(theta,sigma);
	return rand(importDist);
end

function project(theta)
	theta = round(theta,3);
	theta_proj = Variable(k);
	p = minimize(norm_2(theta_proj-theta))
	p.constraints += [abs(theta_proj) <=0.5];
	solve!(p)
	if(p.status == :Error)
		#println(string("Errortheta: ",theta));
	end
	return theta_proj.value
end





numSteps = 1000000

currGuess = 0; 


#termination condition 
#if over M samples condition is met is then break
within_tol_num = 0;
M = 3500; 
temp_theta = 10*ones(k);

#
#epsilon = 0.3;

##another termination condition with true_theta taken when run with 5*10^6 iteration
true_theta = [0.2213373425757975,0.26146134113900443,0.25923947794995156,0.18914835229821317,0.24623283379614955,0.22993840995118123,0.2658631682854538,0.20385800793626835,0.24931816384848166,0.1995707911772092,0.17640811815986274,0.24386883878587773,0.24478578667365214,0.27855255874909424,0.23383200465161366,0.11630341952678153,0.21256577156642883,0.18727380610018446,0.16512252382783685,0.22090198347590376,0.1017832899032087,0.16911869726448892,0.17829109198217927,0.12855322371726513,0.16721648827023436,0.1685856050406186,0.20184749117363665,0.22175443029952066,0.10526008367241982,0.11855078218770156,0.1307581500231431,0.1551888745630051,0.09807078798706756,0.14444540844171505,0.1897217166618541,0.11203938298375081,0.1364378795386616,0.12689702085266016,0.09467313389149294,0.06723128968746235,0.07030153480267481,0.059903553451026346,0.08394146534566002,0.05507140582837172,0.09310303483205398,0.10281781301633314,0.07314629467566691,0.0711534473586929,0.03053444267613992,0.015208972979756003,0.050098338100417195,0.09705698134218087,0.05326668442967701,0.020904859058539473,0.06134573271325995,-0.012968899574494116,0.06953654925038846,-0.019295633905636742,-0.00042060191572459776,-0.028448195166206003,0.015364250457982802,-0.03766828329174628,-0.030599496891112292,0.03269223566679064];


##termination condition for true asset
true_asset = 4.02;

curr_step = 0; 


#projection method or truncated gradient method
#true = TG

test_1 = true; 

times = Float64[];

#numStepArray = logspace(0,5,100);
#numStepArray = [500,1000,2000,4000,8000,16000, 32000, 64000];


errorArray = Float64[];


#=
for numSteps in numStepArray
	#initial theta
	theta = ones(k);

	numAverages = 100;
	avg = 0.0;
	for z = 1:numAverages
		for i in 1:numSteps
			x = generateSample(theta);
			currGuess = currGuess + phi(x)*pdfTargetDist(x)/pdfImportanceDist(x,theta);

			gradient = (theta - x)*phi(x)^2*pdfTargetDist(x)^2/pdfImportanceDist(x,theta)^2 

			alpha = 0.1;
			theta = theta  - (alpha/(alpha*norm(gradient) + 1))*gradient;
		end
		avg = avg + norm(theta - true_theta)/norm(true_theta);
	end
	println(avg/numAverages);
	push!(errorArray,avg/numAverages);
end

println(errorArray);
loglog(numStepArray, errorArray);
title("ADAMC")
xlabel("number of steps");
ylabel("relative error");
print("Hit <enter> to continue")
readline();

#println(curr_step);
#println(currGuess/curr_step);

=#

#initial theta
theta = ones(k);

numSteps = 10^5;

for i = 1:numSteps
	x = generateSample(theta);
	currGuess = currGuess + phi(x)*pdfTargetDist(x)/pdfImportanceDist(x,theta);

	gradient = (theta - x)*phi(x)^2*pdfTargetDist(x)^2/pdfImportanceDist(x,theta)^2 

	#works for sigma = 0.3
	alpha = 1e-4;
	#alpha = 1e-3;
	theta = theta  - (alpha/(alpha*norm(gradient) + 1))*gradient;
	#theta = theta - alpha*gradient;
end

println(string("TSGD estimate" , currGuess/numSteps));
println(string("TSGD theta: ", theta));

plot(theta);
readline();


#=
#initial theta
theta = ones(k);

for i = 1:numSteps
	x = generateSample(theta);
	currGuess = currGuess + phi(x)*pdfTargetDist(x)/pdfImportanceDist(x,theta);

	gradient = (theta - x)*phi(x)^2*pdfTargetDist(x)^2/pdfImportanceDist(x,theta)^2 

	theta = theta  - 0.01/sqrt(i)*gradient;
	theta = project(theta);
	theta = vec(theta);
end

println(string("ADAMC estimate" , currGuess/numSteps));
println(string("ADAMC theta: ", theta));
plot(theta);
readline();
=#


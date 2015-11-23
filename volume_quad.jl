using Convex
using ECOS
using Distributions
using PyPlot

solver = ECOSSolver(verbose = false)
set_default_solver(solver);

# computes the density at x of a gaussian with parameters m and S
function natural_gaussian(m,S,x)
	return (1/(2*pi))*exp(transpose(m)*x - 0.5*trace(S*x*transpose(x)))[1]*exp(-0.5*(transpose(m)*inv(S)*m-logdet(S)))[1]
end

#projects the point m in R^2 onto [0,25]^2
function project_1(m)
	## because otherwise ECOS fails..
	m = round(m,2);
	m_proj = Variable(2,1);
	p = minimize(norm(m-m_proj))
	p.constraints += [m_proj >= 0; m_proj <=25];
	solve!(p)
	if(p.status == :Error)
		println(string("Errorm: ",m));
	end
	return m_proj.value
end


#projects the point S in S^2 with eigenvalues between 1 and 50
function project_2(S)
	## because otherwise ECOS fails..
	S = round(S,2);

	S_proj = Variable(2,2);
	p = minimize(norm_2(S_proj-S))
	p.constraints += [S_proj[1] == S_proj[4]; S_proj[2] == S_proj[3]];
	p.constraints += [S_proj >= eye(2); S_proj <= 50*eye(2)];
	solve!(p)
	if(p.status == :Error)
		println(string("ErrorS: ",S));
	end
	return S_proj.value
end

# returns 1 if x in R^2 is in the quadrilateral Q 
# with corners (0.05,0.9), (0.8,0.9), (1,0.7), (0.15,0.7)
function inQ(x, scalingFactor)
	x1 = (0.05*scalingFactor,0.9*scalingFactor);
	x2 = (0.8*scalingFactor,0.9*scalingFactor);
	x3 = (1*scalingFactor,0.7*scalingFactor);
	x4 = (0.15*scalingFactor,0.7*scalingFactor);
	quad = [x1;x2;x3;x4];
	return InsidePolygon(quad, x);
end

#adapted from solution2 bbs.darthmouth.edu
#a point is inside the polygon if he sum of the angles between the test point and 
# verticies is 2pi 
#assuming that polygon is an array of ordered tuples
#returns 1 if in polygon, 0 otherwise 
function InsidePolygon(polygon, point)
   angle=0;

   ##number of verticies
   n = length(polygon);

   for i in 1:length(polygon)
   		point2 = polygon[i];
   		p1x = point2[1] - point[1];
   		p1y = point2[2] - point[2];


   		#get next vertex in polygon
   		z = i+1;
   		if(z > n)
   			z = 1
   		end

   		point3 = polygon[z];
   		p2x = point3[1] - point[1];
   		p2y =  point3[2] - point[2];

   		angle = angle + Angle2D(p1x,p1y,p2x,p2y);
   end

   if (abs(angle) < pi)
      return 0; 
   else
      return 1;
  	end
  end

##
##   Return the angle between two vectors on a plane
##   The angle is from vector 1 to vector 2, positive anticlockwise
##   The result is between -pi -> pi
##
function Angle2D(x1, y1, x2, y2)

   theta1 = atan2(y1,x1);
   theta2 = atan2(y2,x2);
   dtheta = theta2 - theta1;
   while (dtheta > pi)
      dtheta -= 2*pi;
   end
   while (dtheta < -pi)
      dtheta += 2*pi;
   end

   return(dtheta);
end



#returns 1 if x is in [0,1]^2
function unitSquare(x)
	if x[1] < 0 
		return 0;
	elseif x[1] > 1
		return 0
	elseif x[2] < 0
		return 0
	elseif x[2] >1
		return 0
	end
	
	return 1;
end



#generates a sample from the normal distribution with parameters m and S 
function generateSample(m,S)
#first convert to regular parameters
	Sigma = inv(S);
	mu = Sigma*m; 
	mu = reshape(mu,prod(size(mu)))
	d = MvNormal(mu,Sigma);
	return rand(d);
end



#for truncated gradient
alpha = 0.5

#statistics for mean/variance/rate of convergence
scalingFactor = 1; 
numSteps = 1000000

trueMean = scalingFactor^2*0.16;


total = 0; 
total2 = 0; 

currVal = Float64[];

test_1 = false; 


##very far from minimum
S_vfar = [50 0; 0 50];
m_vfar = [30;30];

#far from minimum 
S_far = [0.5 0; 0 .5];
m_far = [0;0];

### near minimum 
S_true =[13 0; 0 35];
m_true = [7; 28];



#sample starting values
S = S_true;
C = 0.5;
m = m_true;

mean_dist = sqrt(norm(S_true - S)^2 + norm(m_true -m)^2);
println(string("Mean starting distance: ", mean_dist));


numSteps = 10000;

for i in 1:numSteps
	#generate a sample from the current distribution
	x = generateSample(m,S); 

	inQval = inQ(x, scalingFactor);

	temp = (unitSquare(x)*inQval)/natural_gaussian(m,S,x);

	total = total + temp;


	currentGuess = total/i; 
	relativeError = abs(currentGuess - trueMean)/abs(trueMean);

	push!(currVal, relativeError);

	temp2 = (unitSquare(x)^2*inQval)/natural_gaussian(m,S,x)^2

	total2 = total2 + temp2;


	#update values
	prob = natural_gaussian(m,S,x)^2;
	invS = inv(S);


	#update m
	grad_m = (inQval)/(prob) * (invS*m - x);

	if(test_1)
	#vanilla
		temp_m = m - C/sqrt(i)*grad_m;
		#temp_m = project_1(temp_m);
	else
	#truncated gradient
		temp_m = m - alpha*grad_m/(1 + alpha*norm(grad_m));
	end

	#update S 
	grad_S = (inQval)/(2*prob)* (x*transpose(x) - invS*m*transpose(m)*invS - invS);

	if(test_1)
		temp_S = S - C/sqrt(i)*grad_S;
		temp_S = project_2(temp_S);
	else
		#truncated gradient
		temp_S = S - alpha*grad_S/(1 + alpha*norm(grad_S));
	end

	S = temp_S; 
	m = temp_m; 
end


Sigma = inv(S);
mu = Sigma*m; 

println(string("Sigma:"), Sigma);
println(string("mu: "), mu);
println(string("S:"), S);
println(string("m:"), m);

println(string("relative error: "),  currVal[numSteps]);

sampleMean = total/numSteps;
sampleVariance = (total2 - trueMean^2)/numSteps;

println(string("ADAMC Mean: ",sampleMean, " Variance: ", sampleVariance));

##generate sample variance of distribution 

numSteps = 600000; 

for i in 1:numSteps
#generate a sample from the current distribution
	x = generateSample(m,S); 
	inQval = inQ(x,scalingFactor);
	total = total + (unitSquare(x)*inQval)/natural_gaussian(m,S,x);

	temp2 = (unitSquare(x)^2*inQval)/natural_gaussian(m,S,x)^2;

	total2 = total2 + (unitSquare(x)*inQval/natural_gaussian(m,S,x)  - trueMean)^2;
end

sampleMean = total/numSteps;
sampleVariance = total2/numSteps;
println(string("Importance Sampling Mean: ",sampleMean, " Variance: ", sampleVariance));

#plot(1:numSteps, currVal, color="blue", linewidth=0.5, linestyle="-", label = "AdapativeMC");


#=
## importance sampling with gaussian chosen by hand 
total = 0; 
total2 = 0; 

mu = [0.5; 0.8];
Sigma = [0.05 0; 0 0.05];

m = inv(Sigma)*mu; 
S = inv(Sigma);


currVal = Float64[];


for i in 1:numSteps
	#generate a sample from the current distribution
	x = generateSample(m,S); 
	total = total + (unitSquare(x)*inQ(x))/natural_gaussian(m,S,x);

	currentGuess = total/i; 
	relativeError = abs(currentGuess - trueMean)/abs(trueMean);

	push!(currVal, relativeError);

	total2 = total2 + ( (unitSquare(x)*inQ(x))/(natural_gaussian(m,S,x))  - trueMean)^2
end


sampleMean = total/numSteps;
sampleVariance = total2/numSteps;
println(string("Importance Sampling Mean: ",sampleMean, " Variance: ", sampleVariance));

plot(1:numSteps, currVal, color="blue", linewidth=0.5, linestyle="-", label = "importance sample");
=#


##monte carlo method: 
##
#=
total = 0; 
total2 = 0; 

currVal = Float64[];

for i in 1:numSteps
	#take a sample in [0,1]^2
	x = rand(2,1);
	z = inQ(x, scalingFactor);
	total = total + z;

	currentGuess = total/i; 
	relativeError = abs(currentGuess - trueMean)/abs(trueMean);

	push!(currVal, relativeError);
	total2 = total2 + (z - trueMean)^2; 
end

sampleMean = total/numSteps;
sampleVariance = total2/numSteps;


#
println(string("Classical Monte Carlo Mean: ",sampleMean, " Variance: ", sampleVariance));

plot(1:numSteps, currVal, color="red", linewidth=0.5, linestyle="-", label = "Classical Monteo Carlo");

plot(1:numSteps, 1./((1:numSteps).^(1/2)), color = "black", linewidth = 0.5, label = "1/sqrt(n)")
legend(fancybox="true")


ax = gca() # Get the handle of the current axis
ax[:set_yscale]("log"); # Set the y axis to a logarithmic scale
ax[:set_xscale]("log"); # Set the y axis to a logarithmic scale
title("Sampling Rates of Convergence");
xlabel("numSteps");
ylabel("relative Error");

print("Hit <enter> to continue")
readline()
## 
=#


##confidence intervals 
#=

#monte carlo confidence intervals
total = 0; 
total2 = 0; 

means = Float64[];
vars = Float64[];
srand(1234);


for i in 1:numSteps
	#take a sample in [0,1]^2
	x = rand(2,1);
	z = inQ(x);
	total = total + z;
	currMean = total/i; 

	total2 = total2 + (z - currMean)^2; 
	currVar = total2/(i+1);

	push!(vars, currVar);

	push!(means,  currMean);
end

sampleMean = total/numSteps;
sampleVariance = total2/(numSteps-1);

error = 1.96*sqrt(vars)./(1:numSteps).^(1/2);


#
println(string("Classical Monte Carlo Mean: ",sampleMean, " Variance: ", sampleVariance));


plot(1:numSteps, means + error, linestyle = "--", color = "blue");
plot(1:numSteps, means - error, linestyle = "--", color = "blue");
plot(1:numSteps, means, color = "red");


std = 1.96*sampleVariance

ax = gca() # Get the handle of the current axis
ax[:set_xscale]("log"); # Set the y axis to a logarithmic scale
title("Classic Monte Carlo");
xlabel("numSteps");
ylabel("sample mean");
print("Hit <enter> to continue")
readline();
clf();

# importance sampling confidence intervals
total = 0; 
total2 = 0; 



means = Float64[];
vars = Float64[];
srand(444);



for i in 1:numSteps
	#println(string("Current iteration: ", i));

	#generate a sample from the current distribution
	x = generateSample(m,S); 


	#update current mean estimate
	total = total + (unitSquare(x)*inQ(x))/natural_gaussian(m,S,x);
	currMean = total/i; 
	push!(means, currMean);


	#update current variance estimate 
	total2 = total2 + ( (unitSquare(x)*inQ(x))/(natural_gaussian(m,S,x))  - trueMean)^2
	currVar = total2/i;
	push!(vars, currVar);



	#update values
	prob = natural_gaussian(m,S,x)^2;
	invS = inv(S);


	#update m
	temp_m = m - (C*inQ(x))/(prob *sqrt(i)) * (invS*m - x);
	temp_m = project_1(temp_m);

	#update S 
	temp_S = S - (C*inQ(x))/(2*prob*sqrt(i))* (x*transpose(x) - invS*m*transpose(m)*invS - invS)
	S = project_2(temp_S);
	
	m = temp_m; 
end

sampleMean = total/numSteps;
sampleVariance = total2/(numSteps);

error = 1.96*sqrt(vars)./(1:numSteps).^(1/2);


#
println(string("Adaptive Monte Carlo Mean: ",sampleMean, " Variance: ", sampleVariance));
#plot(1:numSteps, means, color="red", linewidth=0.5, linestyle="-", label = "Classical Monteo Carlo");



plot(1:numSteps, means + error, linestyle = "--", color = "blue");
plot(1:numSteps, means - error, linestyle = "--", color = "blue");
plot(1:numSteps, means, color = "red");
ax = gca() # Get the handle of the current axis
ax[:set_xscale]("log"); # Set the y axis to a logarithmic scale
title("Adaptive Monte Carlo");
xlabel("numSteps");
ylabel("sample mean");
print("Hit <enter> to continue")
readline()
=#

## average sample variance versus size of quadrilateral
#=

numSteps = 1000;
variance = Float64[]; 
numScaleSteps = 8;
numVarSteps = 25; 

for z in 1:numScaleSteps
	scalingFactor = 1/z; 
	trueMean = scalingFactor^2*0.16;
	#get average sample variance in 100 runs for this size of quadrilateral
	avgVar = 0; 
	for k in 1:numVarSteps
		total = 0;
		for i in 1:numSteps
			#take a sample in [0,1]^2
			x = rand(2,1);
			z = inQ(x, scalingFactor);
			currentGuess = total/i; 
			total += (z - trueMean)^2; 
		end
		sampleVariance = total/numSteps;
		avgVar += sampleVariance;
	end
	push!(variance, avgVar/numVarSteps)
end

plot(0.16./(1:numScaleSteps).^(2), variance, label = "classic monte carlo")
title("Classic Monte Carlo with 500 steps");
xlabel("quad area");
ylabel("average sample variance over 10 experiments");



#for adaptive MC
variance = Float64[]; 


for z in 1:numScaleSteps
	scalingFactor = 1/z; 
	trueMean = scalingFactor^2*0.16;
	#get average sample variance in 100 runs for this size of quadrilateral
	avgVar = 0; 
	println(string("current scaling iteration: ", z))
	for k in 1:numVarSteps
		total2 = 0; 
		for i in 1:numSteps
			#generate a sample from the current distribution
			x = generateSample(m,S); 

			#update values
			prob = natural_gaussian(m,S,x)^2;
			invS = inv(S);

			#update current variance estimate 
			total2 = total2 + ( (unitSquare(x)*inQ(x, scalingFactor))/(natural_gaussian(m,S,x))  - trueMean)^2

			#update m
			temp_m = m - (C*inQ(x, scalingFactor))/(prob *sqrt(i)) * (invS*m - x);
			temp_m = project_1(temp_m);

			#update S 
			temp_S = S - (C*inQ(x, scalingFactor))/(2*prob*sqrt(i))* (x*transpose(x) - invS*m*transpose(m)*invS - invS)
			S = project_2(temp_S);
			
			m = temp_m; 
		end
		sampleVariance = total2/(numSteps);
		avgVar += sampleVariance;
	end
	push!(variance, avgVar/numVarSteps)
end

plot(0.16./(1:numScaleSteps).^(2), variance, label = "adaptive monte carlo")
title("Monte Carlo with 1000 steps");
xlabel("quad area");
ylabel("average sample variance over 10 experiments");
legend(fancybox="true")
print("Hit <enter> to continue")
readline()


# plot relative error/ sqrt(n)
=#



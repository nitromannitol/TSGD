using Distributions 
using Regression
import Regression:solve, Options


N = 1000; 
p = 3; 
rho = 0;


#We generate data with N observations and p predictors with each
#pair of predictors (x_j, x_i ) having the same population correlation rho
data = []; 
cov = rho*ones(p,p);
cov = cov + (1 - rho)*eye(p,p);
distribution = MvNormal(cov);


#each column in data corresponds to a feature and each row corresponds to a data sample 
data = rand(distribution, N);
data = transpose(data);


#=
#construct the coefficients to have alternating signs and to be exponentially decreasing
true_theta = Float64[];
for i = 1:p
	push!(true_theta,(-1)^i*exp(-2*(i-1)/20));
end
=#
true_theta = [2,-4,1];

#generate the probabilities for each data sample
eta = data*true_theta;
probs = 1./(1 + exp(-eta));

y = Float64[];
#generate bernouli outcomes 
for prob in probs
	b = Bernoulli(prob);
	push!(y,rand(b));
end


#=

##Test TSGD
#parameters 
curr_theta = zeros(p);
temp_theta = 10*ones(p);
epsilon = 10.0^(-10);
alpha = 0.1;
numSteps = 0; 

tic();
while(norm(curr_theta - true_theta) > epsilon)
	numSteps = numSteps + 1; 
	println(norm(curr_theta - true_theta));
	p_vector = transpose(1./(1 + exp(-transpose(curr_theta)*transpose(data))));
	SG = transpose(data)*(y-p_vector);
	temp_theta = curr_theta;
	#curr_theta = curr_theta + alpha*SG/(alpha*norm(SG,Inf)+1);
	curr_theta = curr_theta + (alpha/numSteps)*SG;
end
toc();
println(string("num steps: "), numSteps);
println(norm(curr_theta - true_theta));
println(transpose(curr_theta));
println(true_theta);

=#



#=

println("Implicit Method")

#compare to implicit method

curr_theta = zeros(p);
temp_theta = 10*ones(p);
epsilon = 10.0^(-10);
alpha = 0.001;
numSteps = 0; 

tic();
while(norm(curr_theta - temp_theta) > epsilon)
	p_vector = transpose(1./(1 + exp(-transpose(curr_theta)*transpose(data))));
	r = alpha*( y - p_vector);
	#compute search bounds
	bounds = [0,r];
	if(r < 0)
		bounds = [r,0];
	end
	#solve one dimensional equation by numerical root finding method 

	curr_theta = curr_theta + 

	if(r < 0)
		bo
	bounds = [0,]

	println(norm(curr_theta - true_theta));

	r = alpha*()


	SG = transpose(data)*(y-p_vector);
	temp_theta = curr_theta;
	curr_theta = curr_theta + alpha*SG/(alpha*norm(SG)+1);
	numSteps = numSteps + 1; 
end
toc();
println(string("num steps: "), numSteps);
println(norm(curr_theta - true_theta));
println(string("vec: difference: ", curr_theta - true_theta));
println(true_theta);
=#


#=
println("Newton's Method")

#compare to newton method

##Test TSGD
#parameters 
curr_theta = zeros(p);
temp_theta = 10*ones(p);
epsilon = 10.0^(-10);
alpha = 0.0001;
numSteps = 0; 

tic();
while(norm(curr_theta - temp_theta) > epsilon)
	println(norm(curr_theta - true_theta));
	p_vector = transpose(1./(1 + exp(-transpose(curr_theta)*transpose(data))));
	SG = transpose(data)*(y-p_vector);
	temp_theta = curr_theta;
	curr_theta = curr_theta + alpha*SG/(alpha*norm(SG)+1);
	#curr_theta = curr_theta + alpha*SG;
	numSteps = numSteps + 1; 
end
toc();
println(string("num steps: "), numSteps);
println(norm(curr_theta - true_theta));
println(string("vec: difference: ", curr_theta - true_theta));
println(true_theta);

=#

#calculate the MSE 
z = data*curr_theta;
probs = 1./(1 + exp(-z));
predict = Float64[];
#generate bernouli outcomes 
for prob in probs
	b = Bernoulli(prob);
	push!(predict,rand(b));
end

println(string("MSE: ", sum((y - predict).^2)/N));


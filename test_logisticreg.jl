using Distributions 

using Regression
import Regression: solve, Options


N = 1000; 
p = 5; 
rho = 0;


#We generate data with N observations and p predictors with each
#pair of predictors (x_j, x_i ) having the same population correlation rho
X = []; 
cov = rho*ones(p,p);
cov = cov + (1 - rho)*eye(p,p);
distribution = MvNormal(cov);


#each column in data corresponds to a feature and each row corresponds to a data sample 
X = rand(distribution, N);



#construct the coefficients to have alternating signs and to be exponentially decreasing
true_theta = Float64[];
for i = 1:p
	push!(true_theta,(-1)^i*exp(-2*(i-1)/20));
end

#generate the probabilities for each data sample
z = transpose(X)*true_theta;
probs = 1./(1 + exp(-z));

y = Float64[];
#generate bernouli outcomes 
for prob in probs
	b = Bernoulli(prob);
	push!(y,rand(b));
end

### y[find(x->(x==0),y)] = -1;

tic();
ret = solve(logisticreg(X, y), solver = BFGS());
toc();

println()
# Print results
w_e = ret.sol

#@printf("corr(truth, estimated) = %.6f\n", dot(true_theta, w_e) / (vecnorm(true_theta) * vecnorm(w_e)));

#println(string("package theta_final: ",w_e));
println(string("package Error: ", norm(w_e - true_theta)));


#calculate the MSE 
z = transpose(X)*w_e;
probs = 1./(1 + exp(-z));
predict = Float64[];
#generate bernouli outcomes 
for prob in probs
	b = Bernoulli(prob);
	push!(predict,rand(b));
end

#println(string("package MSE: ", sum((y - predict).^2)/N));



##Test TSGD
#parameters 
curr_theta = zeros(p);
temp_theta = 10*ones(p);
epsilon = 10.0^(-8);
alpha = 0.00001;
numSteps = 0; 

tic();
while(norm(curr_theta - temp_theta) > epsilon)
	#println(norm(curr_theta - temp_theta));
	numSteps = numSteps + 1; 
	#println(norm(curr_theta - true_theta));
	p_vector = transpose(1./(1 + exp(-transpose(curr_theta)*X)));
	SG = X*(y-p_vector);
	temp_theta = curr_theta;
	curr_theta = curr_theta + alpha*SG/(alpha*norm(SG,2)+1);
	#curr_theta = curr_theta + (1/numSteps)*SG;
end
toc();
#println(string("myalgo numsteps: "), numSteps);
println(string("myalgo Error: ", norm(curr_theta - true_theta)));
#println(string("myalgo theta_final: ", transpose(curr_theta)));




#calculate the MSE 
z = transpose(X)*curr_theta;
probs = 1./(1 + exp(-z));
predict = Float64[];
#generate bernouli outcomes 
for prob in probs
	b = Bernoulli(prob);
	push!(predict,rand(b));
end

#println(string("myalgo MSE: ", sum((y - predict).^2)/N));



########################################################################################
println("Implicit Method")

########################################################################################
#compare to implicit method

using Optim

curr_theta = zeros(p);
temp_theta = 10*ones(p);
epsilon = 10.0^(-10);
alpha = 5;
numSteps = 0; 






tic();
while(numSteps <10^3)
	temp_theta =curr_theta;
	numSteps = numSteps+1;
	curr_i = numSteps%N +1;
	curr_x = X[:,curr_i];
	curr_y =y[curr_i];
	a_n = alpha/numSteps;

	pp = transpose(1./(1 + exp(-transpose(curr_theta)*curr_x)));
	r = a_n*(curr_y - pp);
	r = r[1]; 
	#compute search bounds
	bounds = [0,r];
	if(r < 0)
		bounds = [r,0];
	end
	fn(u) = (u - a_n*(curr_y - 1./(1 + exp(transpose(curr_theta)*curr_x) + sum(curr_x.^2)*u)))[1];

	#solve one dimensional equation by numerical root finding method 
	zz = optimize(fn,bounds[1],bounds[2]);
	kk = zz.minimum; 

	curr_theta = curr_theta + kk*curr_x; 
	#println(norm(curr_theta - true_theta));

end
toc();
println(string("implicitalg numsteps: "), numSteps);
println(string("implicitalg Error: ", norm(curr_theta - true_theta)));
#println(string("implicitalg theta_final: ", transpose(curr_theta)));


#println(string("true theta: ", true_theta));



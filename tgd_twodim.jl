using Distributions
d = Normal();



function grad(vector)
	w = vector[1]
	x = vector[2];
	y = vector[3];
	z = vector[4];

	#convex function gradient with minimum at (0,0)
	#return [w^3; x^3; y^3; z^3];
	#return [w^3; x^5; y^7; z^9];
	#return [sign(w); sign(x); sign(y); sign(z)];
	#return [exp(x) - 1; exp(y) - 1; exp(z) - 1];
	return [sinh(w), sinh(x), sinh(y), sinh(z)];
end

function stochastic_grad(vector)
	return grad(vector) + 0*rand(d,4);
	#return grad(vector)
end

x_0 = [20;20;20; 20];
nsteps = 1e6;

for i in 1:nsteps
	#alpha = 1/i; 
	alpha = 1e-3;
	gradval = stochastic_grad(x_0);
	x_0 = x_0 - alpha*gradval;
	#if(alpha*gradval[1] < 1)
	#	x_0 = x_0 - alpha*gradval;
	#else
	#	x_0 = x_0 - alpha*gradval/norm(gradval);
	#end

	#x_0 = x_0 - alpha*gradval/max(1, alpha*norm(gradval));
	#x_0 = x_0 - gradval/norm(gradval);
	#x_0 = x_0 - alpha*gradval/(1 + alpha*norm(gradval));
	#x_0 = x_0 - alpha*gradval/norm(gradval);
	#println(norm(x_0 - [1;1]));
	#println(x_0);
end
#error = norm(x_0 - [1;1]);
error = norm(x_0);
println(string("final error: ",error, " final point: ", x_0));

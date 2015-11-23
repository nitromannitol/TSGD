using Distributions
d = Normal();



function grad(vector)
	x = vector[1];
	y = vector[2];
	return [2*x^2; 750*y^2];
end

function stochastic_grad(vector)
	return grad(vector) + (1/20)*rand(d,2);
	#return grad(vector)
end



curr_x = [10000;10000];
temp_x = [11;11];
##step size
#alpha = 0.5; 
epsilon = 0.0001;
i = 0;
test_1 = false;

while(norm(curr_x - temp_x,2) > epsilon)
	i=i+1;
	alpha = 1/sqrt(i);
	temp_x = curr_x;
	if(test_1)
		curr_x = temp_x - alpha*stochastic_grad(temp_x);
	else
		curr_x = temp_x - alpha*stochastic_grad(temp_x)/(1 + alpha*norm(stochastic_grad(temp_x),2));
	end
	println(curr_x);
end



println(curr_x);
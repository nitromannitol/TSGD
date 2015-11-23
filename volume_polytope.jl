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
function inQ(x, scalingFactor = 1)
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
# this is f(x)
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

function getSampleVariance(m,S)
	numSamples = 10^6;
	seq = Float64[];
	for i=1:numSamples
		x = generateSample(m,S);
		phix = inQ(x);
		fx = unitSquare(x);
		ftheta = natural_gaussian(m,S,x);
		val = phix*fx/ftheta;
		push!(seq, val[1]);
	end
	return var(seq);
end


##very far from minimum
S_vfar = [50 0; 0 50];
m_vfar = [30;30];

#far from minimum 
S_far = [0.5 0; 0 .5];
m_far = [0;0];

### near minimum 
S_true =[13 0; 0 35];
m_true = [7; 28];


#parameter
C = 0.5;
alpha = 1e-3;

#sample starting values
S = S_far;
m = m_far;

println(inv(S));
println(inv(S)*m);

numSteps = 1e6; 

tic();
for i =1:numSteps
	alpha = 1/i;
	x = generateSample(m,S); 

	#cache values to use for gradient_m and S update 
	inQval = inQ(x);
	invS = inv(S);
	prob = natural_gaussian(m,S,x)^2;


	grad_m = (inQval)/(prob) * (invS*m - x);
	grad_S = (inQval)/(2*prob)* (x*transpose(x) - invS*m*transpose(m)*invS - invS);

	m = m - alpha*grad_m/(max(alpha^2*norm(grad_m),1));
	S = S - alpha*grad_S/(max(alpha^2*norm(grad_S),1));
end
toc();

TSGD_var = getSampleVariance(m,S);
Sigma = inv(S);
mu = Sigma*m; 

println(string("m:", m));
println(string("S:", S));
println(string("TSGD_Sigma:"), Sigma);
println(string("TSGD_mu: "), mu);
println(string("TSGD_opt_var: ", TSGD_var));

S = S_far;
m = m_far;


tic();
for i = 1:numSteps
	x = generateSample(m,S); 

	#cache values to use for gradient_m and S update 
	inQval = inQ(x);
	invS = inv(S);
	prob = natural_gaussian(m,S,x)^2;


	grad_m = (inQval)/(prob) * (invS*m - x);
	grad_S = (inQval)/(2*prob)* (x*transpose(x) - invS*m*transpose(m)*invS - invS);

	#ADAMC update
	m = m - C/sqrt(i)*grad_m;
	S = S - C/sqrt(i)*grad_S;

	m = project_1(m);
	S = project_2(S);
end
toc();

ADAMC_var = getSampleVariance(m,S);
Sigma = inv(S);
mu = Sigma*m; 
println(string("ADAMC_opt_var: ", ADAMC_var));
println(string("ADAMC_Sigma:"), Sigma);
println(string("ADAMC_mu: "), mu);




## matrix logarithm function for symmetric positive (AKA diagonalizable) matrices

function logm (A)

egvals, V = eig(A);
if(length(find(egvals.==0))>=1)
  print(" Warning: Principal matrix logarithm is not defined for A with nonpositive real eigenvalues. A non-principal matrix logarithm is returned.\n")
end
V_inv = inv(V);
A2 = V_inv * A * V;

return V*diagm(log(diag(A2)))*V_inv;

end
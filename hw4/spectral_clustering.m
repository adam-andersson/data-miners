% Importing comma-separated edge list in Matlab
E = readmatrix('example1.dat');

% Converting Edge list to the adjacency matrix
col1 = E(:,1);
col2 = E(:,2);
max_ids = max(max(col1,col2));
As = sparse(col1, col2, 1, max_ids, max_ids); 
A = full(As)

% Show sparsity pattern of A
figure(1)
spy(A)

% Converting adjacency matrix to Affinity matrix (weighted)




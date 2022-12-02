% Compute the eigenvectors and eigenvalues of a set of graphs 
% and find out how many communities these graphs have. 

% Importing comma-separated edge list in Matlab
E = readmatrix('example0.dat');

% Converting Edge list to the adjacency matrix
col1 = E(:,1);
col2 = E(:,2);
max_ids = max(max(col1,col2));
As = sparse(col1, col2, 1, max_ids, max_ids); 
A = full(As);

% Getting the eigenvalues
[v, D] = eig(A);

% Find and sort the fiedler vector
[mx,ind] = maxk(D,2);
second_smallest_index = ind(2);
sorted_fiedler_v = sort(v(:,second_smallest_index))

% TODO: Figure out how to plot this thing..



% Importing comma-separated edge list in Matlab
file_path = 'example1.dat'

E = readmatrix(file_path);
% k = 4

for k = 2:5
    A = find_clusters(E, k, file_path);
end

function A = find_clusters(E, k, file_title)
    sgtitle(file_title)

    % Converting Edge list to the adjacency matrix
    col1 = E(:,1);
    col2 = E(:,2);
    max_ids = max(max(col1,col2));
    As = sparse(col1, col2, 1, max_ids, max_ids); 
    A = full(As);
    dim_A = length(A);
    
    % Show sparsity pattern of A
    % spy(A)
    
    
    % Generate the Diagonal Matrix D
    A_sum = sum(A,2);
    D = diag(A_sum);
    
    
    % Calculate the matrix L
    % L = D^(-1/2) A D^(-1/2)
    D_squared = D^(-0.5);
    L = D_squared * A * D_squared;
    
    
    % Finding the eigenvalues and eigenvectors of L
    [V, ~] = eigs(L, k, 'la'); % la = Largest Algebraic
    
    
    % Creating the matrix Y which has Xs rows normalized
    X = V; % rename to adhere to paper
    Y = normr(X);
    
    
    % k-means clustering to partition into k clusters
    idx = kmeans(Y, k);
    
    
    % Plotting the graph with nodes of different clusters highlighted
    subplot(2, 2, k-1)

    colors = ["red", "green", "blue", "yellow", "magenta", "cyan"];  
    G = plot(graph(A));
    
    for i = 1:k
        points_in_cluster = find(idx==i);
        color_index = mod(i, length(colors)) + 1;
        highlight(G, points_in_cluster, 'NodeColor', colors(color_index));
    end

    title(strcat('k=', num2str(k)))

end
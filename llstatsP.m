%
function [L,s] = llstatsP(Y, K, rho, burn_in)
    % Computes log-likelihood with L1 penalty on Kraus operators
    % Inputs:
    %   Y: Observation sequence (subset of training data)
    %   K: Cell array of Kraus operators
    %   rho: Density matrix
    %   burn_in: Number of burn-in steps
    % Outputs:
    %   L: Log-likelihood minus L1 penalty (for compatibility with minimization)
    
    lambda = 5; % Regularization parameter (tune as needed)
    [L, s] = llstats(Y, K, rho, burn_in); % Compute log-likelihood
    % Compute L1 penalty on Kraus operators
    l1_penalty = 0;
    for i = 1:size(K, 1) % Loop over observables
        for j = 1:size(K, 2) % Loop over operators per observable
            l1_penalty = l1_penalty + sum(abs(K{i,j}(:))); % Sum absolute values of all elements
        end
    end
    L = L - lambda * l1_penalty; % Subtract L1 penalty
end
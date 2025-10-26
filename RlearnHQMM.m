function bestK = RlearnHQMM(state_dim, func, Y_train, Y_val, K_learned, rho_learned, burn_in, batch_size, iter, num_batches, k, num_corrupt, solver)
    % RlearnHQMM: Learn HQMM parameters with optional robust filtering for corrupted data
    % Inputs:
    %   state_dim: Dimension of hidden states
    %   func: Likelihood function
    %   Y_train: Training data (NUM_TR_DATA x SEQ_LENGTH matrix of observations)
    %   Y_val: Validation data
    %   K_learned: Initial Kraus operators (cell array)
    %   rho_learned: Initial density matrix
    %   burn_in: Burn-in steps
    %   batch_size: Size of mini-batches
    %   iter: Number of iterations
    %   num_batches: Number of batches
    %   k: Number of resampling iterations
    %   num_corrupt: Number of corrupted sequences to filter (if > 0)
    %   solver: String specifying the optimization solver ('fmincon' or 'patternsearch')
    % Output:
    %   bestK: Optimized Kraus operators
    %
    % Example usage:
    %   bestK = RlearnHQMM(..., 'fmincon');
    %   bestK = RlearnHQMM(..., 'patternsearch');

    % Input validation for solver
    if ~ischar(solver) && ~isstring(solver)
        error('solver must be a string: ''fmincon'' or ''patternsearch''');
    end

    %% Initialization
    bestVal = -Inf;
    num_observables = size(K_learned, 1);
    operators_per_observable = size(K_learned, 2);
    hist = zeros(num_batches, 1);

    % If num_corrupt > 0, apply robust filtering to Y_train
    if num_corrupt > 0 && num_corrupt < size(Y_train, 1)
        train_data_corrupt = Y_train;
        Y_train = remove_corrupted_rows_entropy(train_data_corrupt, num_observables, num_corrupt);
    end

    % Parameter bounds
    gammalbnd = [-pi; -pi; -pi; -pi];
    gammaubnd = [pi; pi; pi; pi];

    % Set up solver options
    options = optimset('Display', 'off'); % Options for fmincon
    ps_options = optimoptions('patternsearch', ...
                             'Display', 'off', ...
                             'UseCompletePoll', true, ...
                             'UseCompleteSearch', true); % Options for patternsearch

    % Select the solver function handle outside the loops
    switch lower(solver)
        case 'fmincon'
            solverFunc = @(x0) fmincon(@findParams, x0, [], [], [], [], ...
                                       gammalbnd, gammaubnd, [], options);
        case 'patternsearch'
            solverFunc = @(x0) patternsearch(@findParams, x0, [], [], [], [], ...
                                            gammalbnd, gammaubnd, [], ps_options);
        otherwise
            error('Unknown solver: %s. Use ''fmincon'' or ''patternsearch''', solver);
    end

    for d = 1:num_batches
        fprintf('Processing batch %i of %i\n', d, num_batches);
        samples = randsample(size(Y_train, 1), batch_size, false); % without replacement

        for it = 1:iter
            %% Generate k proposals
            log_likelihoods = zeros(1, k); 
            K_learneds = cell(1, k);

            for i = 1:k
                %% Update Kraus operators in one proposal
                ops = randsample(1:num_observables*state_dim*operators_per_observable, 2, false);
                op1 = ceil(min(ops)/(state_dim*operators_per_observable));
                mat1 = ceil((min(ops)-(op1-1)*state_dim*operators_per_observable)/state_dim);
                row1 = min(ops) - (((op1-1)*state_dim*operators_per_observable) + (mat1-1)*state_dim);
                op2 = ceil(max(ops)/(state_dim*operators_per_observable));
                mat2 = ceil((max(ops)-(op2-1)*state_dim*operators_per_observable)/state_dim);
                row2 = max(ops) - (((op2-1)*state_dim*operators_per_observable) + (mat2-1)*state_dim);

                % Try initial condition at zero
                gamma = [0; 0; 0; 0]; % denote [phi; psi; delta; theta]
                bestLL = findParams(gamma);

                % Optimize using the selected solver
                gamma = solverFunc(gamma);
                tempLL = findParams(gamma);

                if tempLL < bestLL
                    bestLL = tempLL;
                    K_learned = update(gamma);
                end

                % Store results
                log_likelihoods(i) = bestLL;
                K_learneds{i} = K_learned;
            end

            %% Compute weights based on log-likelihoods
            weights = exp(-log_likelihoods - min(-log_likelihoods)); 
            weights = weights / sum(weights);

            %% Resample one proposal based on weights
            selected_idx = randsample(1:k, 1, true, weights);
            bestLL = log_likelihoods(selected_idx);
            K_learned = K_learneds{selected_idx};

            %% Final update for the iteration
            fprintf('\tNew Log Likelihood: %f\n\n', -bestLL);
        end
        
        % Track validation likelihood
        hist(d) = func(Y_val, K_learned, rho_learned, burn_in);
        if hist(d) > bestVal
            bestVal = hist(d); 
            bestK = K_learned;
        end    
    end
               
    %% Nested function: objective wrapper
    function L = findParams(gamma)
        newK = update(gamma);
        L = func(Y_train(samples, :), newK, rho_learned, burn_in);
        L = -L; % minimize negative likelihood
    end

    %% Nested function: update Kraus operators
    function newK = update(gamma)
        newK = K_learned;
        newK{op1, mat1}(row1, :) = (exp(1i*gamma(1)/2)*exp(1i*gamma(2))*cos(gamma(4))) * K_learned{op1, mat1}(row1, :) + ...
                                   (exp(1i*gamma(1)/2)*exp(1i*gamma(3))*sin(gamma(4))) * K_learned{op2, mat2}(row2, :);

        newK{op2, mat2}(row2, :) = (-exp(1i*gamma(1)/2)*exp(-1i*gamma(3))*sin(gamma(4))) * K_learned{op1, mat1}(row1, :) + ...
                                   (exp(1i*gamma(1)/2)*exp(-1i*gamma(2))*cos(gamma(4))) * K_learned{op2, mat2}(row2, :);
    end    
end
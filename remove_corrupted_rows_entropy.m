% detect_corrupted_rows_entropy.m
% Purpose: Identify and remove corrupted rows in a dataset using entropy and unique value counts
% to detect outliers, even when mean and variance appear normal.

function Y_train = remove_corrupted_rows_entropy(Y_train, OBS, num_corrupt)
    % Input:
    %   Y_train - Numeric matrix (rows x columns) to check for corrupted rows
    %   num_corrupt - Number of corrupted rows to remove
    % Output:
    %   Y_train - Filtered matrix with non-corrupted rows

    if num_corrupt > 0 && num_corrupt < size(Y_train, 1)
        NUM_TR_DATA = size(Y_train, 1);

        % Copy original training data
        Y_train_corrupt = Y_train;

        % Initialize arrays for entropy and unique value counts
        entropies = zeros(NUM_TR_DATA, 1);
        unique_counts = zeros(NUM_TR_DATA, 1);

        % Compute entropy and unique value count for each row
        for i = 1:NUM_TR_DATA
            seq = Y_train_corrupt(i, :);
            
            % Compute frequency of each value (1 to 6)
            counts = histcounts(seq, 0.5:1:OBS+0.5); % Bins for integers 1 to 6
            probs = counts / sum(counts); % Normalize to probabilities
            probs = probs(probs > 0); % Avoid log(0) for entropy
            entropies(i) = -sum(probs .* log2(probs)); % Shannon entropy
            
            % Count unique values in the row
            unique_counts(i) = length(unique(seq));
        end
        
        % Normalize entropy and unique counts to compute z-scores
        z_entropy = (entropies - mean(entropies)) / std(entropies);
        z_unique = (unique_counts - mean(unique_counts)) / std(unique_counts);
        
        % Outlier score: Combine entropy and unique counts
        % Lower entropy or fewer unique values -> higher outlier score
        outlier_scores = -z_entropy - z_unique; % Negative because lower is worse
        
        % Display outlier scores for debugging
        % fprintf('Outlier scores (higher indicates more anomalous):\n');
        % disp(outlier_scores);
        
        % Sort by outlier score (descending) to identify most anomalous rows
        [~, sorted_idx] = sort(outlier_scores, 'descend');
        
        % Keep indices of non-corrupted rows
        keep_indices = sorted_idx(num_corrupt+1:NUM_TR_DATA);
        
        % Display kept indices
        % fprintf('The value of keep_indices is:\n');
        % disp(keep_indices');
        
        % Filter dataset to keep non-corrupted rows
        Y_train = Y_train_corrupt(keep_indices, :);
    else
        fprintf('No rows removed: num_corrupt is %d, dataset size is %d.\n', ...
                num_corrupt, size(Y_train, 1));
    end
end


function [part] = make_xval_partition(n, n_folds)
    %{ 
    Randomly generates a partitioning for n datapoints into n_folds equally
    sized folds (or as close to equal as possible). PART is a 1 X N vector,
    where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
    of the i'th data point.
    %}

    % Create a vector of the values 1 to n_folds repeated.
    part = repmat(1:n_folds, 1, ceil(n/n_folds));

    % Truncate off excess digits.
    part = part(1:n);

    % Randomly permute the values in part. 
    part = part(randperm(n));
end

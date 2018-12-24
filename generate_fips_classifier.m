function mdl = generate_fips_classifier(data, pca_to_keep, dim_subspace, num_learners)
    %{ 
    This method generates a subspace-discriminant ensemble 
    classifier to predict the FIPS code of an observation. 

    Training data is provided through data. The first column
    should be the two-digit FIPS code. The next 2021 columns
    should be the remaining features. 

    While any hyper-parameters can be input, the leaderboard
    submission had the following options, which optimized
    accuracy under given space and time requirements.
    %}

    if nargin < 2
        pca_to_keep = 300;
        dim_subspace = 299;
        num_learners = 2;
    end

    % Format the data for training.
    features = data(:, 2:end);
    fips = data(:, 1);
    
    % PCA.
    pca_to_keep = min(size(features, 2), pca_to_keep);
    [coeffs, scores, ~, ~, ~, centers] = pca(features, ...
        'NumComponents', pca_to_keep);
   
    % Fit the model
    dim_subspace = max(1, min(dim_subspace, size(features, 2) - 1));
    num_learners = max(1, num_learners);
    ensemble = fitcensemble(scores, fips, ...
        'Method', 'Subspace', ...
        'NumLearningCycles', num_learners, ...
        'Learners', 'discriminant', ...
        'NPredToSample', dim_subspace, ...
        'ClassNames', unique(fips));

    % Generate output struct
    mdl.predictFcn = @(x) predict(ensemble, (x - centers) * coeffs);
    mdl.PCACenters = centers;
    mdl.PCACoefficients = coeffs;
    mdl.ClassificationEnsemble = ensemble;
end

function pred_labels = predict_labels(Xtrain, Ytrain, Xtest)
    %{
    Given training data for X and Y and a test set, this
    method makes Y predictions. It does so by first adding
    the state average or predicted state average of each
    observation. It then cross-validates the optimal 
    number of principal components to use in PLS regression
    on a column-by-column basis. 
    %}

    % Append state average features.
    [Xtrain, Xtest] = add_state_avgs(Xtrain, Xtest);
  
    % 5-fold CV. 
    K = 5;
    % Num of output features.
    NF = size(Ytrain, 2); 
    % The first pass values to test for PLS. 
    % In previous testing, optimal value in this range for all columns.
    NC = [1 5*(1:10)]; 
    
    % For each output feature, opt contains the minimum CV error of 
    % that particular column, and the number of principal components
    % that generated that error. 
    opt = zeros(NF, 2);
    
    % For each output feature (plsregress needs a vector target)
    for feat = 1:NF
        % Tracks the cv error of the corresponding number of pc's.
        nc_err = 1:size(NC, 2);
        for nc = 1:size(NC, 2)
            
            [feat, NC(nc)] % print an update
            
            % Perform cross-validation
            indices = make_xval_partition(size(Xtrain, 1), K);
            cv_err = 1:K;
            % run cross validation
            for i = 1:K
                cv_Xtest = Xtrain(indices == i, :); 
                cv_Xtrain = Xtrain(indices ~= i, :);
                cv_Ytest = Ytrain(indices == i, feat); 
                cv_Ytrain = Ytrain(indices ~= i, feat);

                [~,~,~,~,BETA] = plsregress(cv_Xtrain, cv_Ytrain, NC(nc));
               
                cv_yhat = [ones(size(cv_Xtest, 1),1) cv_Xtest]*BETA;
                cv_err(i) = column_error(cv_yhat, cv_Ytest, feat);
            end
            % Take cv error for that number of components.
            nc_err(nc) = mean(cv_err);
        end
        % Take min error (for debugging) and index (for final prediction).
        [v, i] = min(nc_err);
        opt(feat, :) = [v, i];
    end
    
    % Using opt, predict y.
    pred_labels = zeros(size(Xtest, 1), size(Ytrain, 2));
    for feat = 1:NF
        [~,~,~,~,BETA] = plsregress(Xtrain, Ytrain(:, feat), NC(opt(feat, 2)));
        pred_labels(:, feat) = [ones(size(Xtest, 1),1) Xtest]*BETA;
    end
end


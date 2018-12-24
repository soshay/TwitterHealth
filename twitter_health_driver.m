function test_err = twitter_health_driver(filename)
    %{
    This method return the test error of the
    twitter health model. The supplied filename
    must contain all data (training and testing, 
    X and Y). It must be 2031 columns wide, where 
    columns represent features and rows represent 
    observations. The first column must be the 
    five-digit FIPS code, and the final nine 
    columns must be our targe values.
    %}

    holdout = 0.10;
    
    all_data = csvread(filename);
    training_size = floor((1 - holdout)*size(all_data, 1));
    % Exclude final nine values (Y) from X.
    Xtrain = all_data(1:training_size, 1:end - 9);
    % Truncate FIPS code to represent state code only.
    Xtrain(:, 1) = floor(Xtrain(:, 1) / 1000);
    % Take target values for Y
    Ytrain = all_data(1:training_size, end - 8:end);
    % Take held-out observations for testing.
    % We strip FIPS from Xtest for contest restraints. 
    % See README.txt for more information.
    Xtest = all_data(training_size + 1:end, 2:end - 9);
    Ytest = all_data(training_size + 1:end, end - 8:end);
    
    % Make prediction and compute error.
    yhat = predict_labels(Xtrain, Ytrain, Xtest);
    test_err = error_metric(yhat, Ytest);
end






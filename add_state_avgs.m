function [Xtrain, Xtest] = add_state_avgs(Xtrain, Xtest)
    %{
    This method extends the X training and testing data to 
    include the average Y values for all observations of the 
    state that the current observations resides in.
    If the FIPS is known, it simply appends the 
    average for the known state. If the FIPS is
    not known, it predicts the FIPS using a ensemble
    classifier.
    %}

    avg_filename = 'state_avgs.csv';

    % Predict FIPS for Xtest.
    mdl = generate_fips_classifier(Xtrain);
    fips = mdl.predictFcn(Xtest);
    Xtest = [fips Xtest];
    
    % Open saved state averages.
    
    % If not already generated, un-comment the following line:
    % generate_state_avgs('training_data.csv', avg_filename);
    
    state_avgs = csvread(avg_filename);
    
    % Xtrain
    to_append = ones(size(Xtrain, 1), size(state_avgs, 2));
    fips = Xtrain(:, 1);
    for i = 1:size(Xtrain, 1)
        to_append(i, :) = state_avgs(fips(i),:);
    end
    Xtrain = [Xtrain to_append];
    
    % Xtest
    to_append = ones(size(Xtest, 1), size(state_avgs, 2));
    fips = Xtest(:, 1);
    for i = 1:size(Xtest, 1)
        to_append(i, :) = state_avgs(fips(i),:);
    end
    Xtest = [Xtest to_append];
    
end


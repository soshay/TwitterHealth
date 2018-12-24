# TwitterHealth
In short: this is an award-winning model that predicts health outcomes given Twitter LDA frequencies.

In long: this project was developed in response to a graduate-level machine learning competition at the University of Pennsylvania.

The data in training_data.csv was provided as training data for the competition. Within are 1019 observations of county-level Twitter habits. These observations include the following: the FIPS code of the county in question, twenty-one demographic features of the county, and 2000 LDA frequences corresponding to the topics that that county had been tweeting about. The target of the challenge was to predict nine real-valued health outcomes given these demographic and LDA features.

Each competitor's submitted model would then be given a ~1000 observation test set, which, unfortunately, did not include the FIPS code. According to both the hosts of the competition, as well as most of the other competitions, the FIPS was unnecessary as it is a nominal value that contains no inherent information. Final models for the competition had to be under 50MB and run in under ten minutes.

This project uses PLS regression to predict the health outcomes given the demographic and LDA features. However, the insight of this model (and the reason why it ultimately won) was because it engineers nine additional features for each observation. The features were found through close analysis of the FIPS code.

While the five-digit FIPS code seems to include no information, the first two digits are, in fact, the state that that county resides in. In the data set, there were fifty-one unique FIPS codes (fifty states plus D.C.). 

This model uses the training data (in which FIPS codes are provided) to train a subspace-discriminant ensemble classifier that can predict a FIPS code given the non-FIPS demographic and LDA features. Through careful tuning of hyper-parameters, the accuracy of this classifier reached nearly 95%. This model then groups the training data by state, finding the average value of each of the nine health outcomes for each state. It then appends the average health outcomes of the predicted FIPS to each observation. Then, the augmented data is simply passed to PLS regression.

By adding these additional features, overall error was decreased by over 33%. The efficiency of this method is also shown by the final results of the competition: the distance between first and second place was the same distance between second and twenty-first place.

To execute the full model, load all files into MATLAB and call:

twitter_health_driver('training_data.csv');

This will run the model and return its test error. 

To make a prediction, load all files into MATLAB and call:

yhat = predict_labels(Xtrain, Ytrain, Xtest);

Information about how to format these matrices can be found in the predict_labels and twitter_health_driver methods.

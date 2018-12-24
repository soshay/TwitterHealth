function error=column_error(predicted_labels,true_labels, index)
    %{ 
    This function calculates average of normalized RMSE of all labels.
    First input is predicted labels, second input is true labels corresponding
    to your predictions. Third is the column index.
    %}
    scale=[571.0000   29.0000    3.3000   12.0000   17.0000   33.0000   24.0000    3.8000   30.0000]; %this is MaxMin of training labels
    error=sum(sqrt(sum((predicted_labels-true_labels).^2)/size(true_labels,1))./scale(index))/size(true_labels,2);
end
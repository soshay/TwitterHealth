function success = generate_state_avgs(in_filename, out_filename)
    %{
    This method saves the state-by-state error 
    to a CSV.

    The supplied file must contain all data 
    (training and testing, X and Y). It must be 
    2031 columns wide, where columns represent 
    features and rows represent observations. 
    The first column must be the five-digit FIPS 
    code, and the final nine columns must be our 
    targe values.
    %}

    % Generate fips
    all_data = csvread(in_filename);
    Y = all_data(:, end - 8:end);
    fips = floor(all_data(:, 1) / 1000);
    
    uniq_fips = unique(fips);
    
    % Pre-allocate target for speed
    state_avgs = zeros(max(uniq_fips), size(Y, 2));
    
    for i = 1:size(uniq_fips,1)
        Y_curr = Y(fips == uniq_fips(i), :);
        state_avgs(uniq_fips(i), :) = mean(Y_curr, 1);
    end
        
    csvwrite(out_filename, state_avgs);
    success = 1;
end


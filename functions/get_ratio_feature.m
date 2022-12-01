function feature_val = get_ratio_feature(energy_1, energy_2)
% Function to get the ratio feature
energy_ratio = energy_1 / energy_2; 
if (energy_ratio > 100)
    energy_ratio = 100;
elseif (isnan(energy_ratio)) % When 0/0, -> 1
    energy_ratio = 1;
end

feature_val = energy_ratio;
end
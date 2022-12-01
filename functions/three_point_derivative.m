function feature_val = three_point_derivative(energy_1, energy_2, energy_3)
%% Function to get the three point derivative
%energy_1:previous 3rd slot's energy
%energy_2:previous 2nd slot's energy
%energy_3:previous slot's energy

feature_val = (energy_1 - 4*energy_2 + 3*energy_3)/2 ;

end
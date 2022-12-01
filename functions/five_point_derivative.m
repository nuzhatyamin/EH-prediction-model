function feature_val = five_point_derivative(energy_1, energy_2, energy_3, energy_4, energy_5)
%% Function to get the five point derivative
%energy_1:previous 4th slot's energy
%energy_2:previous 3rd slot's energy
%energy_3:previous 2nd slot's energy
%energy_4:previous slot's energy
%energy_5:current slot's energy

feature_val = (3*energy_1 - 16*energy_2 + 36*energy_3 - 48*energy_4 + 25*energy_5)/12;


end
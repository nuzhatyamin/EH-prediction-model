function [feature_matrix, actual_energy] = generate_EH_model_features_byminute_data(date_matrix, EH_data_hourly, EH_data_by_minute, ...
    feature_vector_size, warmup_period, active_slots, prediction_horizon)

%This function generates features for prediction of energy

EH_data_minute_derivative = zeros(size(EH_data_by_minute));

%Get 5-pt derivative averages of past 15,30 and 60 min using by_minute data
for min = 6:size(EH_data_by_minute,2)                 
    EH_data_minute_derivative(:,min) = five_point_derivative(EH_data_by_minute(:, min-5), ...
                        EH_data_by_minute(:, min-4), EH_data_by_minute(:, min-3), ...
                        EH_data_by_minute(:, min-2),EH_data_by_minute(:, min-1)); 
end
%Take average 
for hour = 2:24
    train_data_avg_15mins_derivative(:, hour) = mean(EH_data_minute_derivative(:,60*(hour-2)+46:60*(hour-2)+60),2);
    train_data_avg_30mins_derivative(:, hour) = mean(EH_data_minute_derivative(:,60*(hour-2)+31:60*(hour-2)+60),2);
    train_data_avg_60mins_derivative(:, hour) = mean(EH_data_minute_derivative(:,60*(hour-2)+1:60*(hour-2)+60),2);
end

% Month
% day
% slot (5-20)
% prev slot energy (i-1)
% prev slot energy (i-2)
% prev slot energy (i-3)
% prev slot energy (i-4)
% delta between (i-1), (i-2)
% delta between (i-2), (i-3)
% delta between (i-3), (i-4)
% prev 30 minutes' average derivative
% prev day's same slot (d-1)
% prev day's same slot (d-2)
% prev day's same slot (d-3)
% prev day's same slot (d-4)
% delta between (d-1), (d-2)
% delta between (d-2), (d-3)
% delta between (d-3), (d-4)

n_days = size(date_matrix, 1) - warmup_period;

feature_matrix = zeros(n_days*length(active_slots), feature_vector_size);
actual_energy = zeros(n_days*length(active_slots), 1);

%EH_data_hourly_label = label_generator(EH_data_hourly);

for day = warmup_period+1:size(date_matrix, 1) % loop over all days
    curr_date = date_matrix(day, 3);      % Get current date
    curr_month = date_matrix(day, 2);      % Get current date
    for slot = active_slots
        curr_feature_vector = [curr_month, curr_date, slot, ...
            EH_data_hourly(day, slot - 1), ...
            EH_data_hourly(day, slot - 2), ...
            EH_data_hourly(day, slot - 3), ...
            EH_data_hourly(day, slot - 4), ...
            three_point_derivative(EH_data_hourly(day, slot - 3), EH_data_hourly(day, slot - 2), EH_data_hourly(day, slot - 1)), ...
            get_ratio_feature(EH_data_hourly(day, slot - 1), EH_data_hourly(day, slot - 2)), ...
            get_ratio_feature(EH_data_hourly(day, slot - 2), EH_data_hourly(day, slot - 3)), ...
            get_ratio_feature(EH_data_hourly(day, slot - 3), EH_data_hourly(day, slot - 4)), ...
            train_data_avg_15mins_derivative(day, slot), ...
            train_data_avg_30mins_derivative(day, slot), ...
            train_data_avg_60mins_derivative(day, slot), ...
            EH_data_hourly(day - 1, slot), ...
            EH_data_hourly(day - 2, slot), ...
            EH_data_hourly(day - 3, slot)
            ];
        % indexing for feature matrix
        % (day - 1)*length(active_slots) + slot - active_slots(1) + 1
        current_index = (day - warmup_period - 1)*length(active_slots) + slot - active_slots(1) + 1;
        feature_matrix(current_index, :) = curr_feature_vector;
        actual_energy(current_index) = EH_data_hourly(day, slot + prediction_horizon);
    end
end % day = 1:n_days

% slots_length = size(energy_each_hour_fv,2);
% active_slots = repmat(active_slots',size(energy_each_hour_fv,1),1);
% feature_vector =repelem(eh_dates,size(energy_each_hour_fv,2),1);
% energy_each_hour_fv = reshape(energy_each_hour_fv',[size(feature_vector,1),1]);
% feature_vector(:,4) = active_slots;
% feature_vector(:,16) = energy_each_hour_fv;
% 
% %Calculating the previous slots' solar energy column
% for i = 2:size(feature_vector,1)
%     feature_vector(i,5) = feature_vector(i-1,16);
% end
% for i = 3:size(feature_vector,1)
%     feature_vector(i,6) = feature_vector(i-2,16);
% end
% for i = 4:size(feature_vector,1)
%     feature_vector(i,7) = feature_vector(i-3,16);
% end
% for i = 5:size(feature_vector,1)
%     feature_vector(i,8) = feature_vector(i-4,16);
% end
% 
% %Calculating the previous slots' solar energy difference column
% 
% feature_vector(:,9) = feature_vector(:,5)-feature_vector(:,6);
% feature_vector(:,10) = feature_vector(:,6)-feature_vector(:,7);
% feature_vector(:,11) = feature_vector(:,7)-feature_vector(:,8);
% 
% 
% %Calculating the previous days' solar energy columns
% for i = 4*slots_length+1:size(feature_vector,1)
%     feature_vector(i,12) = feature_vector(i-slots_length,16);
%     feature_vector(i,13) = feature_vector(i-2*slots_length,16);
%     feature_vector(i,14) = feature_vector(i-3*slots_length,16);
%     feature_vector(i,15) = feature_vector(i-4*slots_length,16);
% end
% 

end
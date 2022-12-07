%% Select the user
clear variables; close all;
addpath('./functions');

rng(143);
%% Initialize parameters for solar energy data
solar_data_indices.year_val = 1;
solar_data_indices.month_val = 2;
solar_data_indices.day_val = 3;
solar_data_indices.minute_val = 4;
solar_data_indices.irdata_val = 5;
solar_data_indices.zenith_angle_val = 6;
solar_data_indices.azimuth_angle_val = 7;

% Feature vector details
feature_vector.month = 1;
feature_vector.date = 2;
feature_vector.slot = 3;
feature_vector.energy_s1 = 4;
feature_vector.energy_s2 = 5;
feature_vector.energy_s3 = 6;
feature_vector.energy_s4 = 7;
feature_vector.energy_3ptderivative = 8;
feature_vector.ratios12 = 9;
feature_vector.ratios23 = 10;
feature_vector.ratios34 = 11;
feature_vector.avg_15min_derivative = 12;
feature_vector.avg_30min_derivative = 13;
feature_vector.avg_60min_derivative = 14;
feature_vector.energy_d1_s1 = 15;
feature_vector.energy_d2_s1 = 16;
feature_vector.energy_d3_s1 = 17;

feature_vector_size = size( struct2table(feature_vector),  2);
feature_vector.level = feature_vector_size + 1;

%% set date and time zone for solar model. Also load data for 2016
latitude = 39.742;
longitude = -105.18;
elevation = 1828.8;
SiteTimeZone = -7;

location_data.tz = SiteTimeZone;
location_data.latitude = latitude;
location_data.longitude = longitude;
location_data.elevation = elevation;
% load the solar energy data for the location
load 'data/date_time_matrix_all.mat'
date_time_data = date_time_matrix_all;


load('data/date_matrix_all.mat');
%req_month = 1;
req_year = 2015; %Change to 2015 to generate training data
date_req_month_indices = (date_matrix(:,1) == req_year);% & (date_matrix(:,2) == req_month));
eh_dates_2015 = date_matrix(date_req_month_indices, :);

req_year = 2016; %Change to 2015 to generate training data
date_req_month_indices = find((date_matrix(:,1) == req_year));% & (date_matrix(:,2) == req_month));
eh_dates_2016 = date_matrix(date_req_month_indices, :);

%% Load energy data (per minute) for 2 years

data_2015 = load('data/EH_byminute_2015.mat').energy_by_minute; % 2015
data_2016 = load('data/EH_byminute_2016.mat').energy_by_minute; % 2016

% set train and test yeara
train_year = 2015;
test_year = 2016;

%% convert minute data into hourly data
for hour = 1:24
    EH_data_hourly_2015(:,hour) = sum(data_2015(:,60*(hour-1)+1:60*(hour-1)+60),2);
    EH_data_hourly_2016(:,hour) = sum(data_2016(:,60*(hour-1)+1:60*(hour-1)+60),2);
end
    
train_data_by_minute = data_2015;
train_data = EH_data_hourly_2015;
train_date_matrix = eh_dates_2015;
test_data_by_minute = data_2016;
test_data = EH_data_hourly_2016;
test_date_matrix = eh_dates_2016;

% Index of future slot to predict
% prediction_horizon = 0 for +1h prediction, prediction_horizon = 1 for +2h
% prediction and so on.
prediction_horizon = 0;

%Filtering out the slot indexes that yeild no solar energy at any time of year
active_slots = find(any(train_data~=0));

% Define morning and evening slots
morning_slots = [5, 6, 7, 8, 9, 18, 19, 20];
noon_slots = [10, 11, 12, 13, 14, 15, 16, 17];

division_morning = [0.10, 0.60]; %range multiplier for unequal division
division_noon = [0.0625, 0.60]; %range multiplier for unequal division

nn1_size = 4;       %Energy Prediction Neural Network - 'Fitnet'
neural_net_size = [4 8]; %Level Prediction Neural Network - 'Patternnet'
warmup_period = 4;    %Testing will be done from 5th day. 

%% Get the feature vectors for the morning and noon slots (Train and Test)
[feature_matrix_train, actual_train] = generate_EH_model_features_byminute_data(train_date_matrix, ...
    train_data, train_data_by_minute, feature_vector_size, warmup_period, active_slots, prediction_horizon);

%Generating feature matrix Train for morning slots 
req_morning_idx_train = ismember(feature_matrix_train(:,feature_vector.slot), morning_slots);
 
feature_matrix_train_morning = feature_matrix_train(req_morning_idx_train,:);
actual_train_morning = actual_train(req_morning_idx_train,:);

[feature_matrix_train_morning(:,feature_vector.level), morning_boundaries, labels_morning] = label_generator_for_8model_updated(actual_train_morning,...
     division_morning); %Generating labels to make morning models
 
% Generating feature matrix Train for noon slots
 
[feature_matrix_train_noon, actual_train_noon] = generate_EH_model_features_byminute_data(train_date_matrix,...
     train_data, train_data_by_minute, feature_vector_size, warmup_period, noon_slots, prediction_horizon);
[feature_matrix_train_noon(:,feature_vector.level), noon_boundaries, labels_noon] = label_generator_for_8model_updated(actual_train_noon,...
     division_noon); %Generating labels to make noon models

% % test data
% Generating feature matrix Test for morning slots 
[feature_matrix_test, actual_test] = generate_EH_model_features_byminute_data(test_date_matrix, ...
    test_data, test_data_by_minute, feature_vector_size, warmup_period, active_slots, prediction_horizon);
 
req_morning_idx_test = ismember(feature_matrix_test(:,feature_vector.slot), morning_slots);
 
feature_matrix_test_morning = feature_matrix_test(req_morning_idx_test,:);
actual_test_morning = actual_test(req_morning_idx_test,:);

%dividing the feature matrix accoding to 4 labels for both morning and noon slots
feature_matrix_test_morning(:,feature_vector.level) = label_generator_test(actual_test_morning, morning_boundaries); %This label will be used later to predict with decision tree

[feature_matrix_test_noon, actual_test_noon] = generate_EH_model_features_byminute_data(test_date_matrix, ...
    test_data, test_data_by_minute, feature_vector_size, warmup_period, noon_slots, prediction_horizon);
feature_matrix_test_noon(:,feature_vector.level) = label_generator_test(actual_test_noon, noon_boundaries); %Generating labels to make noon models

features_for_level_prediction = [feature_vector.energy_s1:feature_vector.energy_d3_s1];

[networks_morning, dec_tree_morning, pred_error_nn1_train_morning, pred_error_nn1_test_morning] = generate_2level_nn_byminute(feature_matrix_train_morning, actual_train_morning, ...
    feature_matrix_test_morning, actual_test_morning, features_for_level_prediction, labels_morning, nn1_size, neural_net_size);

%NN level prediction
%Get the predicted levels
nn_labels_morning_train = vec2ind(dec_tree_morning(feature_matrix_train_morning(:,features_for_level_prediction)')); % predict(dec_tree_morning, feature_matrix_train_morning(:,1:18));
nn_labels_morning = vec2ind(dec_tree_morning(feature_matrix_test_morning(:,features_for_level_prediction)')); %predict(dec_tree_morning, feature_matrix_test_morning(:,1:18));
 
[networks_noon, dec_tree_noon, pred_error_nn1_train_noon, pred_error_nn1_test_noon] = generate_2level_nn_byminute(feature_matrix_train_noon, actual_train_noon, ...
     feature_matrix_test_noon, actual_test_noon,features_for_level_prediction, labels_noon, nn1_size, neural_net_size);
 
nn_labels_noon_train = vec2ind(dec_tree_noon(feature_matrix_train_noon(:,features_for_level_prediction)'));%predict(dec_tree_noon,feature_matrix_train_noon(:,1:18));
nn_labels_noon = vec2ind(dec_tree_noon(feature_matrix_test_noon(:,features_for_level_prediction)')); %predict(dec_tree_noon,feature_matrix_test_noon(:,1:18));

%Checking the accuracy of the label predictions
NN_labels_accuracy_morning_train = 100-100*sum(ne(feature_matrix_train_morning(:,feature_vector.level),...
     nn_labels_morning_train'))/size(nn_labels_morning_train',1);
NN_labels_accuracy_noon_train = 100-100*sum(ne(feature_matrix_train_noon(:,feature_vector.level),...
     nn_labels_noon_train'))/size(nn_labels_noon_train',1);
 
% Test accuracy
NN_labels_accuracy_morning = 100-100*sum(ne(feature_matrix_test_morning(:,feature_vector.level),...
     nn_labels_morning'))/size(nn_labels_morning',1);
NN_labels_accuracy_noon = 100-100*sum(ne(feature_matrix_test_noon(:,feature_vector.level),...
     nn_labels_noon'))/size(nn_labels_noon',1);


%% Call the manual function for NN inference
%implement_patternnet_manually
%We will update the weights for online learning. Hence, we will extract the
%weights first
% Extract the weights for morning
[W1_morning, W2_morning, W3_morning, offsets_morning] = extract_net_weights_and_offset(dec_tree_morning);
% Extract the weights for noon
[W1_noon, W2_noon, W3_noon, offsets_noon] = extract_net_weights_and_offset(dec_tree_noon);

% % Energy prediction with sequential inputs
% Apply to the noon slots at first
 
% Making one vector for decision tree labels using morning and noon labels
% to predict energy
%We are doing this to make a vector of the morning and noon levels into one
%vector sequentially 
m = 1;
n = 1;
for i = 1:size(feature_matrix_test, 1)
    curr_features = feature_matrix_test(i, :);

    if ismember(curr_features(feature_vector.slot),morning_slots)
        dec_tree_labels(i,:) = nn_labels_morning(m);
        m = m + 1;
    else
        dec_tree_labels(i,:) = nn_labels_noon(n) ; %dec_tree_labels_noon(n);
        n = n + 1;
    end
end
 
%% Main energy prediction loop
%Apply on the test data
data_test_predicted = zeros(size(test_data));
data_test_predicted_adapt = zeros(size(test_data));

morning_counter = 1;
noon_counter = 1;

noon_test_labels = [];

acc_offline_noon  = [];
acc_online_noon = [];

acc_offline_morning  = [];
acc_online_morning = [];

morning_test_labels = [];

morning_counter_nn1 = {1,1,1};
morning_energy_predictions = cell(1, length(labels_morning));       %Energy prediction by nn1 without applying RL

acc_offline_morning_nn1 = cell(1, length(labels_morning));

noon_counter_nn1 = {1,1,1};
noon_energy_predictions = cell(1, length(labels_noon));         %Energy prediction by nn1 without applying RL
acc_offline_noon_nn1 = cell(1, length(labels_noon)); 


actual_morning_counter = 1;
actual_noon_counter = 1;

actual_test_morning_cell = cell(1, length(labels_morning));

N = 20; %No. of examples
buffer_matrix_morning = cell(1, length(labels_morning));
curr_network_cell_morning = cell(1, length(labels_morning));
energy_predictions_by_adapt_morning = cell(1, length(labels_morning));
acc_online_morning_adapt = cell(1, length(labels_morning));
acc_offline_morning_adapt = cell(1, length(labels_morning));

buffer_matrix_noon = cell(1, length(labels_noon));
curr_network_cell_noon = cell(1, length(labels_noon));
energy_predictions_by_adapt_noon = cell(1, length(labels_noon));
acc_online_noon_adapt = cell(1, length(labels_noon));
acc_offline_noon_adapt = cell(1, length(labels_noon));

for curr_label = 1:length(labels_morning)
    curr_net = networks_morning{curr_label};
    curr_networks_cell_morning{:,curr_label} = networks_morning{curr_label};
end

actual_test_noon_cell = cell(1, length(labels_noon));
for curr_label = 1:length(labels_noon)
    curr_net = networks_noon{curr_label};
    curr_networks_cell_noon{:,curr_label} = networks_noon{curr_label};
end
curr_month = 0;
curr_year = 2016;
for i = 1:size(feature_matrix_test, 1)
    curr_features = feature_matrix_test(i, 1:feature_vector_size);

    nn_features = feature_matrix_test(i, features_for_level_prediction);
    if ismember(curr_features(feature_vector.slot),morning_slots)
        [current_label, P, Oin] = implement_patternnet_manually(nn_features',...
            W1_morning, W2_morning, W3_morning, offsets_morning);
        
        curr_net = networks_morning{current_label};
        energy_prediction = curr_net(curr_features');
        morning_energy_predictions{current_label} = [morning_energy_predictions{current_label}; energy_prediction];

        actual_energy = actual_test_morning(actual_morning_counter);
        actual_test_morning_cell{current_label}(morning_counter_nn1{current_label}) = actual_energy;
        
        energy_pred_adapt =  curr_networks_cell_morning{current_label}(curr_features');
        energy_predictions_by_adapt_morning{current_label} = [energy_predictions_by_adapt_morning{current_label}; energy_pred_adapt];
        curr_buffer_entry = [curr_features, actual_energy];
        
        %Set condition for updating buffer matrix
        abs_error = abs(actual_energy - energy_pred_adapt)/actual_energy;
        if (abs_error >= 0)
            if (actual_energy > 0)
                buffer_matrix_morning{current_label} = [buffer_matrix_morning{current_label}; curr_buffer_entry];
            end    
        end
        
        if size(buffer_matrix_morning{current_label},1) == N
            [new_net, y] = adapt(curr_networks_cell_morning{current_label}, buffer_matrix_morning{current_label}(:,1:17)',buffer_matrix_morning{current_label}(:,18)'); 
            buffer_matrix_morning{current_label} = [];
            curr_networks_cell_morning{current_label} = new_net;
        end
       
        offline_acc_nn1= mean(abs(actual_test_morning_cell{current_label}(1:morning_counter_nn1{current_label})' - ...
            morning_energy_predictions{current_label}(1:morning_counter_nn1{current_label})));
        online_acc_adapt = mean(abs(actual_test_morning_cell{current_label}(1:morning_counter_nn1{current_label})' - ...
        energy_predictions_by_adapt_morning{current_label}(1:morning_counter_nn1{current_label})));
        
        acc_offline_morning_nn1{current_label}  = [acc_offline_morning_nn1{current_label};offline_acc_nn1];
        acc_online_morning_adapt{current_label} = [acc_online_morning_adapt{current_label}; online_acc_adapt];
        morning_counter_nn1{current_label} = morning_counter_nn1{current_label} + 1;
        

        actual_morning_counter = actual_morning_counter + 1;
        %RL update part and accuracy calculation
        actual_label = feature_matrix_test_morning(morning_counter,feature_vector.level);
        morning_test_labels = [morning_test_labels; current_label];
        
        offline_acc = 100-100*nnz(feature_matrix_test_morning(1:morning_counter,feature_vector.level) - ...
            nn_labels_morning(1:morning_counter)')/morning_counter;
        online_acc = 100-100*nnz(feature_matrix_test_morning(1:morning_counter,feature_vector.level) - ...
            morning_test_labels(1:morning_counter))/morning_counter;
        
        acc_offline_morning  = [acc_offline_morning;offline_acc];
        acc_online_morning = [acc_online_morning;online_acc];

        morning_counter = morning_counter + 1;
        
    else % Afternoon slot
        
        [current_label, P, Oin] = implement_patternnet_manually(nn_features',...
            W1_noon, W2_noon, W3_noon, offsets_noon);
        curr_net = networks_noon{current_label};
        energy_prediction = curr_net(curr_features');
        actual_energy = actual_test_noon(actual_noon_counter);
        actual_test_noon_cell{current_label}(noon_counter_nn1{current_label}) = actual_energy;
        noon_energy_predictions{current_label} = [noon_energy_predictions{current_label}; energy_prediction];
        
        energy_pred_adapt =  curr_networks_cell_noon{current_label}(curr_features');
        energy_predictions_by_adapt_noon{current_label} = [energy_predictions_by_adapt_noon{current_label}; energy_pred_adapt];
        curr_buffer_entry = [curr_features, actual_energy];
                        
        abs_error = abs(actual_energy - energy_pred_adapt)/actual_energy;
        if (abs_error >= 0)
            if (actual_energy > 0)
                buffer_matrix_noon{current_label} = [buffer_matrix_noon{current_label}; curr_buffer_entry];
            end
        end
        
        if size(buffer_matrix_noon{current_label},1) == N
            
            [new_net,y] = adapt(curr_networks_cell_noon{current_label}, buffer_matrix_noon{current_label}(:,1:17)',buffer_matrix_noon{current_label}(:,18)'); 
            buffer_matrix_noon{current_label} = [];
            curr_networks_cell_noon{current_label} = new_net;
        end
        
        
        offline_acc_nn1 = mean(abs(actual_test_noon_cell{current_label}(1:noon_counter_nn1{current_label})' - ...
            noon_energy_predictions{current_label}(1:noon_counter_nn1{current_label})));
        online_acc_adapt = mean(abs(actual_test_noon_cell{current_label}(1:noon_counter_nn1{current_label})' - ...
        energy_predictions_by_adapt_noon{current_label}(1:noon_counter_nn1{current_label})));
        
        acc_offline_noon_nn1{current_label}  = [acc_offline_noon_nn1{current_label};offline_acc_nn1];
        acc_online_noon_adapt{current_label} = [acc_online_noon_adapt{current_label}; online_acc_adapt];
        noon_counter_nn1{current_label} = noon_counter_nn1{current_label} + 1;
        
        actual_noon_counter = actual_noon_counter + 1;

        % RL update part and accuracy calculation
        actual_label = feature_matrix_test_noon(noon_counter,feature_vector.level);
                 noon_test_labels = [noon_test_labels; current_label];
        
        offline_acc = 100-100*nnz(feature_matrix_test_noon(1:noon_counter,feature_vector.level) - ...
            nn_labels_noon(1:noon_counter)')/noon_counter;
        online_acc = 100-100*nnz(feature_matrix_test_noon(1:noon_counter,feature_vector.level) - ...
            noon_test_labels(1:noon_counter))/noon_counter;
        
        acc_offline_noon  = [acc_offline_noon;offline_acc];
        acc_online_noon = [acc_online_noon;online_acc];
        
        noon_counter = noon_counter + 1;
    end
    
    index_value = (test_date_matrix(:, 1) == curr_year) & ...
            (test_date_matrix(:, 2) == curr_features(1)) & ...
        (test_date_matrix(:, 3) == curr_features(2));
    index_value = find(index_value ~= 0);
    data_test_predicted(index_value, curr_features(feature_vector.slot)) = energy_prediction;
    data_test_predicted_adapt(index_value, curr_features(feature_vector.slot)) = energy_pred_adapt;
    
    if (curr_features(1) == 12 && curr_features(2) == 31 && curr_features(3) == 20)
        curr_year = curr_year + 1;
    end
    curr_month = curr_features(1);
end % i = 1:size(feature_matrix_test, 1)
% 
data_test_predicted(data_test_predicted < 0) = 0;
data_test_predicted_adapt(data_test_predicted_adapt < 0) = 0;
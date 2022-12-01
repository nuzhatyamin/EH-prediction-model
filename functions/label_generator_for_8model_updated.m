function [train_data_label, label_boundaries, labels] = label_generator_for_8model_updated(train_data,range_multiplier)

%% Function to get the label 
% Take the actual energy of the train data and get the range of the data.
% Divide the data into different ranges according to the multiplier and
% make labels

train_data_label = zeros(size(train_data));
range_data = range(train_data, 'all');
label_range = range_data;

label_boundaries = [0];
labels = [1];

for i = 1:length(range_multiplier)
    next_boundary = range_multiplier(i)*label_range;
    label_boundaries = [label_boundaries, next_boundary];
end
label_boundaries = [label_boundaries, label_range];

for i = 1:length(label_boundaries)-1
    idx = find(train_data >= label_boundaries(i) & train_data <= label_boundaries(i+1));
    train_data_label(idx) = labels(i);
    next_label = labels(i)+ 1;
    labels = [labels next_label];
end
labels(end) = [];
%label_boundaries = label_boundaries(2:end-1);
end
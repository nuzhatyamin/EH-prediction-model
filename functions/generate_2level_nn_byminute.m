function [energy_prediction_networks, level_prediction_network, pred_error_train, pred_error_test] = ...
    generate_2level_nn_byminute(feature_matrix_train,actual_train, ...
    feature_matrix_test, actual_test, feature_vector, labels, nn1_size, nn2_size)

%% This function generates two neural networks
% first one is for energy prediction - fitnet
% Second one is for level prediction - patternnet

energy_prediction_networks = [];
% Generate 3 energy prediction networks for each label in the label array
for i = 1:length(labels)
    % Go through each label and get the corresponding feature matrix for
    % each label
    req_label_idx_train = find(feature_matrix_train(:,size(feature_matrix_train,2)) == labels(i));
    feature_matrix_train_label = feature_matrix_train(req_label_idx_train,1:size(feature_matrix_train,2)-1);
    actual_train_label = actual_train(req_label_idx_train,:);
    req_label_idx_test = find(feature_matrix_test(:,size(feature_matrix_test,2)) == labels(i));
    feature_matrix_test_label = feature_matrix_test(req_label_idx_test, 1:size(feature_matrix_test,2)-1);
    actual_test_label = actual_test(req_label_idx_test,:);
    % Perform linear model fitting with constraints    
    % Define lower and upper bounds for the model values. We lower bound by
    % zero to ensure that the EH predictions do not go negative
    net = fitnet(nn1_size);
    net = train(net, feature_matrix_train_label', actual_train_label');
    net.layers{2}.transferFcn='purelin';
    pred_label_train = net(feature_matrix_train_label');
    pred_label_test = net(feature_matrix_test_label');
    pred_error_train(i) = mean(abs(actual_train_label - pred_label_train'));
    pred_error_test(i) = mean(abs(actual_test_label - pred_label_test'));
    energy_prediction_networks{i} = net;
end

% Generate a patternnet model to classify labels

% decision_tree = fitctree(feature_matrix_train(:,1:18),feature_matrix_train(:,19), 'MinLeafSize', 40);
level_prediction_network = patternnet(nn2_size);
labels_decision = ind2vec(feature_matrix_train(:,size(feature_matrix_train,2))');
level_prediction_network = train(level_prediction_network, feature_matrix_train(:,feature_vector)',labels_decision);
end
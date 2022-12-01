function [transition_label,prob_test, output_in] = implement_patternnet_manually(feature_vector, W1,W2,W3, offsets)


feature_vector = normalize_feature(feature_vector, offsets);

% Add bias to layer 1 as the first row and get the hidden layer output
% before activation

m = size(feature_vector, 2);

hidden_layer_1 = ([ones(1, m); feature_vector]' * W1);

hidden_layer_1 = tansig_activation(hidden_layer_1);

% Prepare input for the hidden layer 2 by adding bias
hidden_layer_2_in = [ones(m, 1) hidden_layer_1];
hidden_layer_2_y = hidden_layer_2_in* W2;
hidden_layer_2_y = tansig_activation(hidden_layer_2_y);

% generate the input for output layer
output_in = [ones(m, 1) hidden_layer_2_y];
output_layer_y = (output_in * W3);

% Apply softmax classification
[transition_label, prob_test] = softmax_classification(output_layer_y);

end
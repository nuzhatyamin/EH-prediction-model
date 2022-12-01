
function train_data_label = label_generator_test(data,label_boundaries)

%% This function assigns labels to the test data based on the label_boundaries 

label = 1;

train_data_label = zeros(size(data,1), 1);
for i = 1:length(label_boundaries)-1
    idx = find(data >= label_boundaries(i) & data <= label_boundaries(i+1));
    train_data_label(idx) = label;
    %This step is needed to ensure if any actual energy is more than the
    %range of data, it gets the label
    if (i == length(label_boundaries)-1)
        idx = find(data >= label_boundaries(i+1));
        train_data_label(idx) = label;
    end
    label = label+1;
end

end
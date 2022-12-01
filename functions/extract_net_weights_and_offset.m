function [W1, W2, W3, offsets] = extract_net_weights_and_offset(net)

W1 = net.IW{1};
W2 = net.LW{2};
W3 = net.LW{3,2};
b1 = net.b{1};
b2 = net.b{2};
b3 = net.b{3};

W1 = [b1, W1]';
W2 = [b2, W2]';
W3 = [b3, W3]';

offsets = struct;
offsets.xoffset=net.inputs{1}.processSettings{1}.xoffset; 
offsets.gain=net.inputs{1}.processSettings{1}.gain;
offsets.ymin=net.inputs{1}.processSettings{1}.ymin;

end
function normalized_feature = normalize_feature(feature_vector, offsets)

normalized_feature = (feature_vector - offsets.xoffset).*offsets.gain + offsets.ymin;

end
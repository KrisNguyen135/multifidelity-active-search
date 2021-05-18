function [probs, n, d] = knn_model_fast(problem, train_ind, observed_labels, ...
    test_ind, weights, alpha)

n = nan(numel(test_ind), 2);

for i = 1:2
    n(:, i) = alpha(i) + ...
              sum(weights(test_ind, train_ind(observed_labels == i)), 2);
end

d     = sum(n, 2);
probs = n ./ d;

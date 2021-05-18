function [n, d, p] = update_knn_model(problem, n, d, new_train_ind, ...
    new_label, test_ind, weights)

n(:, new_label) = n(:, new_label) + weights(test_ind, new_train_ind);
d               = d               + weights(test_ind, new_train_ind);

p = n ./ d;

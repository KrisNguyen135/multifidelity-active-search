function [problem, labels_l, labels_h, weights, alpha, nearest_neighbors, similarities] ...
    = load_mf_data(data_name, data_dir, theta)

max_k = 500;
if ~exist('data_dir', 'var')
  data_dir = '../data/';
end

if ~exist('theta', 'var')
  theta = 0.1;
end

switch data_name
case 'citeseer_data'
  if true  % ~contains(data_dir, 'storage1')
    data_dir = fullfile(data_dir, 'citeseer');
  end
  data_path = fullfile(data_dir, data_name);
  load(data_path);
  alpha               = [0.05 1];
  num_points = size(x, 1);
  problem.points      = (1:num_points)';
  problem.num_points  = num_points;

  filename = fullfile(data_dir, 'citeseer_data_nearest_neigbors.mat');

  if exist(filename, 'file')
    load(filename);
  else
    [nearest_neighbors, distances] = ...
      knnsearch(x, x, ...
      'k', max_k + 1);

    save(filename, 'nearest_neighbors', 'distances');
  end

  %% there are duplicates in the data
  % e.g. nearest_neighbors(160, 1:2) = [18, 160]
  % that means x(160,:) and x(18,:) are identical
  for i = 1:num_points
    if nearest_neighbors(i, 1) ~= i
      dup_idx = find(nearest_neighbors(i, 2:end) == i);
      nearest_neighbors(i, 1+dup_idx) = nearest_neighbors(i, 1);
      nearest_neighbors(i, 1) = i;
    end
  end

  % limit to only top k
  k = 50;
  nearest_neighbors = nearest_neighbors(:, 2:(k + 1))';
  % distances = distances(:, 2:(k + 1))';
  similarities = ones(size(nearest_neighbors));

  % precompute sparse weight matrix
  row_index = kron((1:num_points)', ones(k, 1));
  weights = sparse(row_index, nearest_neighbors(:), 1, ... %1 / distances(:), ...
    num_points, num_points);

  % create label vector
  labels_h = 2 * ones(size(x, 1), 1);
  labels_h(connected_labels == 3) = 1;
  problem.num_classes = 2;

case 'citeseer_data_subset'  % a randomly selected quarter of citeseer
    data_dir = fullfile(data_dir, 'citeseer');
    data_path = fullfile(data_dir, 'citeseer_data');
    load(data_path);
    alpha      = [0.05 1];
    num_points = size(x, 1);

    subset           = 0.25;
    selected_ind     = randsample(num_points, fix(num_points * subset));
    x                = x(selected_ind, :);
    connected_labels = connected_labels(selected_ind, :);
    num_points       = size(x, 1);

    problem.points      = (1:num_points)';
    problem.num_points  = num_points;

    filename = fullfile(data_dir, 'citeseer_data_nearest_neigbors_subset.mat');

    if exist(filename, 'file')
      load(filename);
    else
      [nearest_neighbors, distances] = ...
        knnsearch(x, x, ...
        'k', 501);

      save(filename, 'nearest_neighbors', 'distances');
    end

    %% there are duplicates in the data
    % e.g. nearest_neighbors(160, 1:2) = [18, 160]
    % that means x(160,:) and x(18,:) are identical
    for i = 1:num_points
      if nearest_neighbors(i, 1) ~= i
        dup_idx = find(nearest_neighbors(i, 2:end) == i);
        nearest_neighbors(i, 1+dup_idx) = nearest_neighbors(i, 1);
        nearest_neighbors(i, 1) = i;
      end
    end

    % limit to only top k
    k = 25;
    nearest_neighbors = nearest_neighbors(:, 2:(k + 1))';
    % distances = distances(:, 2:(k + 1))';
    similarities = ones(size(nearest_neighbors));

    % precompute sparse weight matrix
    row_index = kron((1:num_points)', ones(k, 1));
    weights = sparse(row_index, nearest_neighbors(:), 1, ... %1 / distances(:), ...
      num_points, num_points);

    % create label vector
    labels_h = 2 * ones(size(x, 1), 1);
    labels_h(connected_labels == 3) = 1;
    problem.num_classes = 2;

case 'bmg_data'
    data_dir  = fullfile(data_dir, 'bmg');
    data_path = fullfile(data_dir, data_name);
    load(data_path);
    % remove labels from features
    x = bmg_data(:, 1:(end - 1));

    % create label vector
    labels_h = 2 * ones(size(x, 1), 1);
    labels_h(bmg_data(:, end) <= 1) = 1;

    % remove rows with nans
    ind = (~any(isnan(x), 2));
    x      =      x(ind, :);
    labels_h = labels_h(ind);

    num_points = size(x, 1);

    train_portion = 0.1;
    rng('default');
    train_ind = crossvalind('holdout', num_points, 1 - train_portion);

    % can be reproduced above
    ind = [1, 33, 39, 45, 46, 53, 111, 135, 141, 165, 185, 200, 201];

    % limit features to those selected
    x = x(~train_ind, ind);
    %     x = x(:, ind);
    num_points = size(x, 1);
    labels_h   = labels_h(~train_ind);

    % remove features with no variance
    x = x(:, std(x) ~= 0);

    % normalize data
    x = bsxfun(@minus, x,     mean(x));
    x = bsxfun(@times, x, 1 ./ std(x));

    problem.points      = x;
    problem.num_classes = 2;
    problem.num_points  = num_points;

    % filename = fullfile(data_dir, 'bmg_nearest_neighbors.mat');
    % filename = fullfile(data_dir, 'bmg_nearest_neigbors.mat');
    filename = fullfile(data_dir, 'new_bmg_nearest_neigbors.mat');

    if exist(filename, 'file')
      load(filename, 'nearest_neighbors', 'distances');
    else
      [nearest_neighbors, distances] = ...
        knnsearch(problem.points, problem.points, ...
        'k', max_k + 1);

      % deal with a small number of ties in dataset
      for i = 1:num_points
        if (nearest_neighbors(i, 1) ~= i)
          ind = find(nearest_neighbors(i, :) == i);
          nearest_neighbors(i, ind) = nearest_neighbors(i, 1);
          nearest_neighbors(i, 1)   = i;
        end
      end

      save(filename, 'nearest_neighbors', 'distances');
    end

    % limit to only top k
    k = 50;
    nearest_neighbors = nearest_neighbors(:, 2:(k + 1))';
    similarities = ones(size(nearest_neighbors));
    % precompute sparse weight matrix
    row_index = kron((1:num_points)', ones(k, 1));
    weights = sparse(row_index, nearest_neighbors(:), 1, ...
      num_points, num_points);

    alpha = [0.05, 1];

otherwise  % drug discovery
  alpha         = [0.001, 1];
  num_inactives = 100000;

  if contains(data_dir, 'storage1')
      data_dir = sprintf('%s/chemicals/', data_dir);
  end

  if contains(data_name, 'ecfp')
    filename = sprintf('target_%s_ecfp4_nearest_neighbors_%d.mat', ...
      data_name(5:end), num_inactives);
    fingerprint = 'ecfp4';
  elseif contains(data_name, 'gpidaph')
    filename = sprintf('target_%s_gpidaph3_nearest_neighbors_%d.mat', ...
      data_name(8:end), num_inactives);
    fingerprint = 'gpidaph3';
  end

  data_dir  = fullfile(data_dir, fingerprint);
  data_path = fullfile(data_dir, filename);
  load(data_path);

  num_points  = size(nearest_neighbors, 1);
  num_actives = num_points - num_inactives;

  labels_h                  = ones(num_points, 1);
  labels_h(1:num_inactives) = 2;

  % Limit to k nearest neighbors.
  k                 = 100;
  nearest_neighbors = nearest_neighbors(:, 1:k)';
  similarities      = similarities(:, 1:k)';

  row_idx = kron((1:num_points)', ones(k, 1));
  weights = sparse(row_idx, nearest_neighbors(:), similarities(:), ...
    num_points, num_points);
end

%% Randomly generate L labels.
rng default;
% Each point has a theta probability of having its label flipped.
% flip_draws = binornd(1, theta, num_points, 1);
% labels_l   = mod(labels_h - 1 + flip_draws, 2) + 1;

% "move"
labels_l = labels_h;

positive_mask = (labels_h == 1);
positive_ind  = find(positive_mask);
negative_ind  = find(~positive_mask);

num_positives = numel(positive_ind);
num_flips     = fix(num_positives * theta);

flip_to_negative = randsample(1:num_positives, num_flips);
labels_l(positive_ind(flip_to_negative)) = 2;

flip_to_positive = randsample(1:(num_points - num_positives), num_flips);
labels_l(negative_ind(flip_to_positive)) = 1;

%% other stats
problem.points            = (1:num_points)';
problem.num_points        = num_points;
problem.num_classes       = 2;
problem.max_num_influence = max(sum(weights > 0, 1));
problem.q                 = 0.5;

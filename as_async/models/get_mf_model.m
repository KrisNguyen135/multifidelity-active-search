function model = get_mf_model(model, varargin)

model = @(problem, train_ind_l, observed_labels_l, ...
          train_ind_h, observed_labels_h, test_ind_h) ...
  model(problem, train_ind_l, observed_labels_l, ...
        train_ind_h, observed_labels_h, test_ind_h, varargin{:});

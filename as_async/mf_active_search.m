% `problem` is a struct with these fields:
% - budget: in how many L queries can be run
% - k: how long an H query takes (in how long an L query takes)
% - time: how much time has passed
% - running_h: the H query that is running (if any)
function [utilities, queries, queried_probs, computed, pruned1, pruned2] = ...
    mf_active_search(problem, train_ind_l, observed_labels_l, train_ind_h, ...
    observed_labels_h, labels_l, labels_h, selector, policy, q_fn, message_prefix)

if ~exist('message_prefix', 'var'), message_prefix = ''; end

verbose = isfield(problem, 'verbose') && problem.verbose;
if verbose
    fprintf('\n%s Initial data: (%d, %d) on L, (%d, %d) on H\n', ...
            message_prefix, train_ind_l, observed_labels_l, ...
            train_ind_h, observed_labels_h);
end

queries       = [];
queried_probs = [];
utilities     = [];

computed = [];
pruned1  = [];
pruned2  = [];

problem.time = 1;
while problem.time <= problem.budget - problem.k + 1
    if verbose
        tic;
        fprintf('\n%s Iteration %d', message_prefix, problem.time);
    end

    test_ind_l = selector(problem, train_ind_l, []);
    test_ind_h = selector(problem, [train_ind_h; problem.running_h], []);

    problem.q = q_fn(problem, train_ind_l, observed_labels_l, ...
                     train_ind_h, observed_labels_h);

    if isempty(test_ind_h)
        if verbose, fprintf('\n'); end
        warning('mf_active_search:no_points_selected', ...
                ['after %d steps, no points were selected. ' ...
                 'Ending run early.'], i);
        return;
    end

    if numel(test_ind_h) == 1 && numel(test_ind_l) == 0
        this_chosen_ind_h = test_ind_h;
    elseif mod(problem.time - 1, problem.k) == 0  % beginning of a period
        [this_chosen_ind_h, this_chosen_ind_l, this_chosen_prob_h, ...
            this_chosen_prob_l, num_computed_h, num_pruned1_h, num_pruned2_h, ...
            num_computed_l, num_pruned1_l, num_pruned2_l] = ...
            policy(problem, train_ind_l, observed_labels_l, train_ind_h, ...
            observed_labels_h, test_ind_l, test_ind_h, true);

        problem.running_h = this_chosen_ind_h;
        queries           = [queries; this_chosen_ind_h];
        queried_probs     = [queried_probs; this_chosen_prob_h];

        computed = [computed; num_computed_h];
        pruned1  = [pruned1;   num_pruned1_h];
        pruned2  = [pruned2;   num_pruned2_h];

        if verbose
            fprintf('\n%s (%d, H) chosen.', message_prefix, this_chosen_ind_h);
        end
    else
        [~, this_chosen_ind_l, ~, this_chosen_prob_l, ~, ~, ~, num_computed_l, ...
            num_pruned1_l, num_pruned2_l] = policy(problem, train_ind_l, ...
            observed_labels_l, train_ind_h, observed_labels_h, test_ind_l, ...
            test_ind_h, false);
    end

    if mod(problem.time, problem.k) == 0
        train_ind_h         = [train_ind_h; problem.running_h];
        this_chosen_label_h = labels_h(problem.running_h);
        observed_labels_h   = [observed_labels_h; this_chosen_label_h];
        if this_chosen_label_h == 1
            utilities = [utilities; 1];
        else
            utilities = [utilities; 0];
        end

        if verbose
            fprintf('\n%s (%d, H) label revealed: %d', ...
                    message_prefix, problem.running_h, this_chosen_label_h);
        end

        problem.running_h = [];
    elseif problem.time == problem.budget - problem.k + 1
        train_ind_h         = [train_ind_h; problem.running_h];
        this_chosen_label_h = labels_h(problem.running_h);
        observed_labels_h   = [observed_labels_h; this_chosen_label_h];
        if this_chosen_label_h == 1
            utilities = [utilities; 1];
        else
            utilities = [utilities; 0];
        end

        if verbose
            fprintf('\n%s (%d, H) label revealed: %d', ...
                    message_prefix, problem.running_h, this_chosen_label_h);
        end

        problem.running_h = [];
        problem.time      = problem.time + 1;
        continue;
    end

    % no need to record utility here
    train_ind_l         = [train_ind_l; this_chosen_ind_l];
    this_chosen_label_l = labels_l(this_chosen_ind_l);
    observed_labels_l   = [observed_labels_l; this_chosen_label_l];
    queries             = [queries; this_chosen_ind_l];
    queried_probs       = [queried_probs; this_chosen_prob_l];

    computed = [computed; num_computed_l];
    pruned1  = [pruned1;   num_pruned1_l];
    pruned2  = [pruned2;   num_pruned2_l];

    if verbose
        fprintf('\n%s (%d, L) chosen, label revealed: %d. Took %.4f seconds\n', ...
                message_prefix, this_chosen_ind_l, this_chosen_label_l, toc);
        fprintf("%s Model's q: %.2f\n", message_prefix, problem.q);
    end

    problem.time = problem.time + 1;
end

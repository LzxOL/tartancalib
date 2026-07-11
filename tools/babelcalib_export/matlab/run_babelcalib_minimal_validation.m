% Minimal BabelCalib validation for TartanCalib-exported multi-board data.
%
% Usage from MATLAB:
%   run('/path/to/tartancalib/tools/babelcalib/run_babelcalib_minimal_validation.m')
%
% Optional variables before running:
%   babelcalib_root = '/Users/linzhaoxian/lzx-ws/project/calibr/babelcalib';
%   export_dir = '/path/to/exported/babelcalib_dataset';
%   output_prefix = '/path/to/results/babelcalib_minimal';

if ~exist('babelcalib_root', 'var')
    babelcalib_root = '/Users/linzhaoxian/lzx-ws/project/calibr/babelcalib';
end
if ~exist('export_dir', 'var')
    export_dir = fullfile(pwd, 'result_may', 'babelcalib_export');
end
if ~exist('output_prefix', 'var')
    output_prefix = fullfile(export_dir, 'minimal_validation');
end

run(fullfile(babelcalib_root, 'init.m'));
run(fullfile(babelcalib_root, 'calib_cfg.m'));

train_data = load(fullfile(export_dir, 'train.mat'), 'corners', 'boards', 'imgsize');
test_data = load(fullfile(export_dir, 'test.mat'), 'corners', 'boards', 'imgsize');

fprintf('Loaded train images: %d\n', numel(train_data.corners));
fprintf('Loaded test images : %d\n', numel(test_data.corners));
fprintf('Boards             : %d\n', numel(train_data.boards));
fprintf('Image size [h w]   : %d %d\n', train_data.imgsize(1), train_data.imgsize(2));

[model, train_res] = calibrate( ...
    train_data.corners, train_data.boards, train_data.imgsize, ...
    cfg{:}, 'save_results', output_prefix);

[test_model, test_res] = get_poses( ...
    model, test_data.corners, train_data.boards, train_data.imgsize, ...
    cfg{:}, 'save_results', output_prefix);

fprintf('Train RMS: %.6f px\n', train_res.rms);
fprintf('Test RMS : %.6f px\n', test_res.rms);


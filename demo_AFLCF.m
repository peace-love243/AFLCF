% This demo script runs the AFLCF tracker with deep features on the

% Add paths
setup_paths();

%  Load video information
base_path  =  './sequences';
%video  = choose_video(base_path);
video = 'Bird1'; %  Singer1  Biker  Bolt  Ironman  Woman   Animal1  BMX4   Bird1 Jumping Jump  Shaking
 
video_path = [base_path '/' video];
[seq, gt_boxes] = load_video_info(video_path);

% Run AFLCF tracker
results = run_AFLCF(seq);

pd_boxes = results.res;
thresholdSetOverlap = 0: 0.05 : 1;
success_num_overlap = zeros(1, numel(thresholdSetOverlap));
res = calcRectInt(gt_boxes, pd_boxes);
for t = 1: length(thresholdSetOverlap)
    success_num_overlap(1, t) = sum(res > thresholdSetOverlap(t));
end
cur_AUC = mean(success_num_overlap) / size(gt_boxes, 1);
FPS_vid = results.fps;
display([video  '---->' '   FPS:   ' num2str(FPS_vid)   '    op:   '   num2str(cur_AUC)]);

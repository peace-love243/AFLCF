function results = tracker(params)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%save results figure
pathDraw = '.\tmp\imgs\';% The folder that will stores the images with overlaid bounding box
pathSave = pathDraw;
%%%%%%%
% Get sequence info
% learning_rate = params.learning;

[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;

% Load learning parameters
admm_max_iterations = params.max_iterations;
init_penalty_factor = params.init_penalty_factor;
max_penalty_factor = params.max_penalty_factor;
penalty_scale_step = params.penalty_scale_step;
temporal_regularization_factor = params.temporal_regularization_factor; 

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2];
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'exact');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = unique(feature_info.data_sz, 'rows', 'stable');
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');
num_feature_blocks = size(feature_sz, 1);

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = mat2cell(feature_sz, ones(1,num_feature_blocks), 2);
filter_sz = feature_sz;
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% Construct the Gaussian label function
yf = cell(numel(num_feature_blocks), 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma_factor = params.output_sigma_factor;
    output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz(i)))) * output_sigma_factor;
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    yf{i}           = fft2(y); 
end
yf0=yf;
% [q,p,t]=size(yf{2})
% imshow(ifft(yf{1}))
% Compute the cosine windows   not work
cos_window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', feature_sz_cell, 'uniformoutput', false);
%%%%%%%%small  center work
% cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
% cos_window = cellfun(@(cos_window) cos_window(2:end-1,2:end-1), cos_window, 'uniformoutput', false);
% Define spatial regularization windows
reg_window = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    reg_scale = floor(base_target_sz/params.feature_downsample_ratio(i));
    use_sz = filter_sz_cell{i};    
    reg_window{i} = ones(use_sz) * params.reg_window_max;
    range = zeros(numel(reg_scale), 2);
    
    % determine the target center and range in the regularization windows
    for j = 1:numel(reg_scale)
        range(j,:) = [0, reg_scale(j) - 1] - floor(reg_scale(j) / 2);
    end
    center = floor((use_sz + 1)/ 2) + mod(use_sz + 1,2);
    range_h = (center(1)+ range(1,1)) : (center(1) + range(1,2));
    range_w = (center(2)+ range(2,1)) : (center(2) + range(2,2));
    
    reg_window{i}(range_h, range_w) = params.reg_window_min;
end

% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';

newton_iterations = params.newton_iterations;

% Use the translation filter to estimate the scale
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;

if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

seq.time = 0;

% Define the learning variables
f_pre_f = cell(num_feature_blocks, 1);
cf_f = cell(num_feature_blocks, 1);
beta = 0;
% Allocate
scores_fs_feat = cell(1,1,num_feature_blocks);
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end

    tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we 
    % just want to initialize the tracker there
    occ=false;
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
            sample_scale = currentScaleFactor*scaleFactors;
            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info,seq.frame);
           
            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false); 
            
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xtf{k1}), 3));
            scores_fs_sum = scores_fs_feat{k1};
%             [a,b,c]=size(scores_fs_sum);
            for k = block_inds
                
                scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf{k}), 3));
                scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
%                 scores_fs_sum = 0.772*scores_fs_sum +  0.228*scores_fs_feat{k};
%                 scores_fs_sum =scores_fs_sum +  scores_fs_feat{k};
                scores_fs_sum1 = scores_fs_feat{k}.*(beta) + scores_fs_sum.*(1-beta);
%                 [e,f,g]=size(scores_fs_sum1)  ;
            end
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs_sum1 = permute(gather(scores_fs_sum1), [1 2 4 3]);
            
            responsef_padded = resizeDFT2(scores_fs_sum, output_sz);       %shallow response
            response = ifft2(responsef_padded, 'symmetric');
            
           responsef_padded2 = resizeDFT2(scores_fs_sum1, output_sz);     %deep response
            response2 = ifft2(responsef_padded2, 'symmetric') ; 
%             response2 = response2 .* beta;
            response3 = response2 + response .* (1-beta);
            responsef_padded3 = fft2(response3);
            [disp_row, disp_col, sind] = resp_newton(response2, responsef_padded2, newton_iterations, ky, kx, output_sz); 
%             [disp_row, disp_col, sind,~, max_response, ~] = resp_newton(squeeze(response)/2, squeeze(response2), newton_iterations, ky, kx, output_sz);    %postion
           
            if seq.frame>2
                frame=seq.frame;

%                 response_shift=circshift(response,[-floor(disp_row) -floor(disp_col)]);
                response_shift=circshift(response2,[-floor(disp_row) -floor(disp_col)]);
                response_pre_shift=circshift(response_pre,[-floor(disp_row_pre) -floor(disp_col_pre)]);
%                 response_diff=abs(abs(response_shift-response_pre_shift)./response_pre_shift);
                response_diff=abs(abs(response_shift-response_pre_shift)./response_pre_shift);
%                 [occ]=updateRefmu(response_diff,frame);
                response_diff=circshift(response_diff,floor(size(response_diff)/2));
              
                response_diff1=abs(abs(response_shift-response_pre_shift));
%                 [a,b,c]=size(response_diff1)
                response_diff1= response_diff1(:, :, sind);
%                 temporal_regularization_factor(1)=max(response_diff1(:))*100;
%                 temporal_regularization_factor(2)=max(response_diff1(:))*100;
                 temporal_regularization_factor1 = norm(response_diff1);
                 temporal_regularization_factor2=max(max(response_diff1));

                 if temporal_regularization_factor1 > 10
                 else
%                      temporal_regularization_factor(1)=10-temporal_regularization_factor1+6;
%                      temporal_regularization_factor(2)=10-temporal_regularization_factor1+6;
%                           temporal_regularization_factor(1)=1/(exp(temporal_regularization_factor1)-1)-2;
%                           temporal_regularization_factor(2)=1/(exp(temporal_regularization_factor1)-1)-2;

                         temporal_regularization_factor(1)=17/(exp(temporal_regularization_factor1+temporal_regularization_factor2)-1);
                          temporal_regularization_factor(2)=17/(exp(temporal_regularization_factor1+temporal_regularization_factor2)-1);
                 end
%                     varience=0.8*max_response(range_h, range_w)+0.2*response_pre(range_h, range_w)-...
%                     0.5*response_diff(range_h, range_w);   %response attention20200704
%                 reg_window(range_h, range_w) = {varience}; 
            end
            % save response in last frame
             response_pre=response2;
            % save translation of response in last frame
            disp_row_pre=disp_row;
            disp_col_pre=disp_col;
        
            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(sind);        
            scale_change_factor = scaleFactors(sind);
            
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            %%%%
            if pos(1)<0||pos(2)<0||pos(1)>size(im,1)||pos(2)>size(im,2)
                pos=old_pos;
            end
            %%%%%%%%%%
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end

            % Update the scale
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            iter = iter + 1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % extract image region for training sample
    sample_pos = round(pos);

    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info,seq.frame);

    % do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);

    % compute the fourier series
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);

     
    % train the CF model for each feature
    for k = 1: numel(xlf)
 
 %%%%%%%%%%
 %sample stragety1  not work
%         if (seq.frame == 1)
%             if (k==1)
%                 model_xf1=xlf{k};
%                  model_xf = model_xf1;
%             else
%                  model_xf2=xlf{k};
%                  model_xf = model_xf2;
%             end
%         else
%              if (k==1)
%                     model_xf1=((1 - learning_rate) * model_xf1) + (learning_rate * xlf{k});
%                  model_xf = model_xf1;
%             else
%                  model_xf2=((1 - learning_rate) * model_xf2) + (learning_rate * xlf{k});
%                  model_xf = model_xf2;
%              end
%         end
%%%%%%%

%sample stragety2
        model_xf = xlf{k};

        if (seq.frame == 1)
            f_pre_f{k} = zeros(size(model_xf));
            mu = 0;
        else
            mu = temporal_regularization_factor(k);
        end
        if ~occ
        % intialize the variables
        f_f = single(zeros(size(model_xf)));
        g_f = f_f;
        h_f = f_f;
        gamma  = init_penalty_factor(k);
        gamma_max = max_penalty_factor(k);
        gamma_scale_step = penalty_scale_step(k);
        
        % use the GPU mode
        if params.use_gpu
            model_xf = gpuArray(model_xf);
            f_f = gpuArray(f_f);
            f_pre_f{k} = gpuArray(f_pre_f{k});
            g_f = gpuArray(g_f);
            h_f = gpuArray(h_f);
            reg_window{k} = gpuArray(reg_window{k});
            yf{k} = gpuArray(yf{k});
        end
%         [a,b,c]=size(yf{k}) 
        
        % pre-compute the variables
        T = prod(output_sz);
        S_xx = sum(conj(model_xf) .* model_xf, 3);
        Sf_pre_f = sum(conj(model_xf) .* f_pre_f{k}, 3);
        Sfx_pre_f = bsxfun(@times, model_xf, Sf_pre_f);

        % solve via ADMM algorithm
        iter = 1;
        while (iter <= admm_max_iterations)

            % subproblem f
            B = S_xx + T * (gamma + mu);
            Sgx_f = sum(conj(model_xf) .* g_f, 3);
            Shx_f = sum(conj(model_xf) .* h_f, 3);
 
            f_f = ((1/(T*(gamma + mu)) * bsxfun(@times,  yf{k}, model_xf)) - ((1/(gamma + mu)) * h_f) +(gamma/(gamma + mu)) * g_f) + (mu/(gamma + mu)) * f_pre_f{k} - ...
                bsxfun(@rdivide,(1/(T*(gamma + mu)) * bsxfun(@times, model_xf, (S_xx .*  yf{k})) + (mu/(gamma + mu)) * Sfx_pre_f - ...
                (1/(gamma + mu))* (bsxfun(@times, model_xf, Shx_f)) +(gamma/(gamma + mu))* (bsxfun(@times, model_xf, Sgx_f))), B);

            %   subproblem g
            g_f = fft2(argmin_g(reg_window{k}, gamma, real(ifft2(gamma * f_f+ h_f)), g_f));

            %   update h
            h_f = h_f + (gamma * (f_f - g_f));

            %   update gamma
            gamma = min(gamma_scale_step * gamma, gamma_max);
            
            iter = iter+1;
        end
        
        % save the trained filters
        f_pre_f{k} = f_f;
        cf_f{k} = f_f;
        end
    end
    
    %%%%%%%feature adptive learning
%     beta = yf{1} +yf{2}
        T1 = prod(output_sz);
     yf0{1} = resizeDFT2( yf0{1}, output_sz); 
     yf0{2} = resizeDFT2( yf0{2}, output_sz);
      yf_final = 0.5*yf0{1} +0.5*yf0{2};
      scores_train_feat1 = gather(sum(bsxfun(@times, conj(cf_f{1}), xlf{1}), 3));
      scores_train_feat2 = gather(sum(bsxfun(@times, conj(cf_f{2}), xlf{2}), 3));
      scores_train_feat2 = resizeDFT2(scores_train_feat2, output_sz);
      score_all = abs(scores_train_feat2-scores_train_feat1);   %%分数差值图
      X= T1*0.001+conj(score_all) .* score_all;     %分母
%       beta = conj(score_all) .*yf_final +  (conj(score_all) .* score_all.*conj(score_all)  .* yf_final) ./ X;
      beta = (conj(score_all) .* ( yf_final-scores_train_feat1-0.5*0.001*T1)) ./ X;
%       beta=mapminmax(beta,0,1); 
      beta1 = 1-ifft2(beta, 'symmetric');
      

    %%%%%%%%%%%%%%%%%%
    
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    seq.time = seq.time + toc();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % visualization
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        
%         imagesc(im_to_show);
%         hold on;
%         rectangle('Position',rect_position_vis, 'EdgeColor','r', 'LineWidth',2);
% %         ground_truth=params.ground_truth(seq.frame,:);
% %         rectangle('Position',ground_truth, 'EdgeColor','g', 'LineWidth',2);
% %         text(10, 10, [int2str(seq.frame) '/'  int2str(size(seq.image_files, 1))], 'color', [0 1 1]);
%         hold off;
%         axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
% %          imwrite(frame2im(getframe(gcf)), [pathSave  num2str(seq.frame) '.png']);  
        
        if seq.frame == 1
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, [int2str(seq.frame) '/'  int2str(size(seq.image_files, 1))], 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            resp_sz = round(img_sample_sz*currentScaleFactor*scaleFactors(sind));
            xs = floor(old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;

%             resp_handle = imagesc(xs, ys, fftshift(y));colormap jet;
%             alpha(resp_handle, 1);
%             beta_handle = imagesc(xs, ys, fftshift(beta1(:,:,1)));colormap jet;
%             alpha(beta_handle, 1);
%             ground_truth=params.ground_truth(seq.frame,:);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
%             rectangle('Position',ground_truth, 'EdgeColor','w', 'LineWidth',2);
%              text(20, 30, ['# Frame : ' int2str(seq.frame) '/' int2str(size(seq.image_files, 1))], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            text(10, 15, ['# Frame : '  int2str(seq.frame) ], 'Color','y', 'FontWeight','bold', 'FontSize',24);
            hold off;
%             imwrite(frame2im(getframe(gcf)), [pathSave  num2str(seq.frame) '.png']);
        end
        
        drawnow
    end
end

[~, results] = get_sequence_results(seq);

disp(['fps: ' num2str(results.fps)])


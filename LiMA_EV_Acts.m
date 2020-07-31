%Extracts feature activations for early vision models (GBJ, GIST)
%Extracts features for both original images, and resized images
%
%Created by Vlad Ayzenberg
%3.30.20

clear all;

addpath('gistdescriptor/')  

exp = {'Exp1', 'Exp2'};
cond = {'SF', 'Size'};

stim = {{'23_Skel', '23_Bulge', '31_Skel', '31_Bulge','26_Skel', '26_Bulge'}, {'31_0_Skel', '31_0_Bulge','31_50_Skel', '31_50_Bulge'}};

skel = {{'23','31', '26'},{'31_0', '31_50'}};
SF = {'Skel', 'Bulge', 'Balloon', 'Shrink', 'Wave'};

imScale = [.1, .2, .3, .4, .5];
imTrans = .2;
frames = 300; %Number of frames

%% GIST Parameters:
clear param
param.imageSize = [256 256]; % it works also with non-square images (use the most common aspect ratio in your set)
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale
param.numberBlocks = 4;
param.fc_prefilt = 4;


for ee = 1:length(exp)
	allData = {};
	n=1;

    for sz = 1:length(imScale);

        for sk = 1:length(skel{ee})
            for sf = 1:length(SF)
                stimActs_GBJ = [];
                sizeActs_GBJ = [];
                stimActs_GIST = [];
                sizeActs_GIST = [];


                st = 1;
                for fn = 1:frames

                    %Load original image
                    ogIM = imread(['Frames/Figure_', skel{ee}{sk},'_',SF{sf}, ...
                        '/Figure_', skel{ee}{sk},'_',SF{sf},'_', int2str(fn), '.jpg']);
                    sizeIM = zeros(round(size(ogIM,1)*(imScale(sz)+1)), round(size(ogIM,1)*(imScale(sz)+1)),3,'uint8');
                    sizeIM(:,:,1) = 119;
                    sizeIM(:,:,2) = 119;
                    sizeIM(:,:,3) = 119;
                    %Overlay OG image on blank in top left corner
                    %This reduces the size and shifts it by imScale %
                    %sizeIM(1:size(ogIM,1), 1:size(ogIM,1), :) = ogIM;
                    padSize = round(((size(ogIM,1)*(imScale(sz)+1))- size(ogIM,1))/2,0);

                    %Resize images by 20%
                    sizeIM = padarray(ogIM, [padSize, padSize], 119); %Pad with 119 (gray) by % size reduction

                    %Resize to GBJ input size
                    ogIM = imresize(ogIM, [256,256]);
                    sizeIM = imresize(sizeIM, [256,256]);

                    %Extract Gabor Magnitudes
                    %ogGBJ = GWTWgrid_Simple(ogIM, 0, 1);
                    sizeGBJ = GWTWgrid_Simple(sizeIM, 0, 1);

                    %stimActs_GBJ(fn,:) = ogGBJ(:)';
                    sizeActs_GBJ(fn,:) = sizeGBJ(:)';

                    %Extract GIST Magnitudes
                    %stimActs_GIST(fn,:) = LMgist(ogIM, '', param);
                    sizeActs_GIST(fn, :) = LMgist(sizeIM, '', param);

                end
                %Save out activations
                save(['Activations/EV_Acts/Figure_', skel{ee}{sk},'_',SF{sf}, '_GBJ_Acts'], 'stimActs_GBJ');
                save(['Activations/EV_Acts/Figure_', skel{ee}{sk},'_',SF{sf}, '_GBJ_Acts_Size', int2str(imScale(sz)*100)], 'sizeActs_GBJ');

                save(['Activations/EV_Acts/Figure_', skel{ee}{sk},'_',SF{sf}, '_GIST_Acts'], 'stimActs_GIST');
                save(['Activations/EV_Acts/Figure_', skel{ee}{sk},'_',SF{sf}, '_GIST_Acts_Size', int2str(imScale(sz)*100)], 'sizeActs_GIST');



                skel{ee}{sk}
                SF{sf}
            end
            
        end

    end
end


				






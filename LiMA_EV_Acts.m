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

        for sk = 1:length(skel{ee})
            for sf = 1:length(SF)
                stimActs_GBJ = [];
                stimActs_GIST = [];

                for fn = 1:frames

                    %Load original image
                    ogIM = imread(['Frames/Figure_', skel{ee}{sk},'_',SF{sf}, ...
                        '/Figure_', skel{ee}{sk},'_',SF{sf},'_', int2str(fn), '.jpg']);
                    
                    %Resize to GBJ input size
                    ogIM = imresize(ogIM, [256,256]);
                    
                    %Extract Gabor Magnitudes
                    ogGBJ = GWTWgrid_Simple(ogIM, 0, 0);
         
                    stimActs_GBJ(fn,:) = ogGBJ(:)';
                    

                    %Extract GIST Magnitudes
                    stimActs_GIST(fn,:) = LMgist(ogIM, '', param);

                end
                %Save out activations
                save(['Activations/EV_Acts/Figure_', skel{ee}{sk},'_',SF{sf}, '_GBJ_Acts'], 'stimActs_GBJ');

                save(['Activations/EV_Acts/Figure_', skel{ee}{sk},'_',SF{sf}, '_GIST_Acts'], 'stimActs_GIST');

                skel{ee}{sk}
                SF{sf}
            end
            
        end

end


				






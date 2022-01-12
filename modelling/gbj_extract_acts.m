%Extracts feature activations for early vision models (GBJ)
%Extracts features for both original images, and resized images
%
%Created by Vlad Ayzenberg
%3.30.20
%updated 1.11.22 for eLife revision
%Testing the branch

clear all;

study_dir = '/user_data/vayzenbe/GitHub_Repos/LiMA';

exp = {'Exp1', 'Exp2'};
cond = {'SF', 'Size'};

stim = {{'23_Skel', '23_Bulge', '31_Skel', '31_Bulge','26_Skel', '26_Bulge'}, {'31_0_Skel', '31_0_Bulge','31_50_Skel', '31_50_Bulge'}};

skel = {{'23','31', '26'},{'31_0', '31_50'}};
SF = {'Skel', 'Bulge'};

frames = 300; %Number of frames


for ee = 1:length(exp)
	allData = {};
	n=1;

        for sk = 1:length(skel{ee})
            for sf = 1:length(SF)
                stimActs_GBJ = [];
                stimActs_GIST = [];

                st = 1;
                for fn = 1:frames

                    %Load original image
                    ogIM = imread(['/user_data/vayzenbe/GitHub_Repos/LiMA/','Frames/Figure_', skel{ee}{sk},'_',SF{sf}, ...
                        '/Figure_', skel{ee}{sk},'_',SF{sf},'_', int2str(fn), '.jpg']);

                    %Resize to GBJ input size
                    ogIM = imresize(ogIM, [256,256]);


                    %Extract Gabor Magnitudes
                    [acts, phase] = GWTWgrid_Simple(ogIM);
                    
         
                    stimActs_GBJ(fn,:) = [acts(:);phase(:)];
                    allData{fn} = cat(3, acts, phase);
                    



                end
                %Save out activations
                save(['gbj_data/Figure_', skel{ee}{sk},'_',SF{sf}, '_gbj_acts'], 'stimActs_GBJ');
                save(['gbj_data/Figure_', skel{ee}{sk},'_',SF{sf}, '_gbj_ims'], 'allData');
                
                

                skel{ee}{sk}
                SF{sf}
                
            end
            
        end

end


				






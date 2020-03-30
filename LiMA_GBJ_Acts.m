clear all;

exp = {'Exp1', 'Exp2'};
cond = {'SF', 'Size'};

stim = {{'23_Skel', '23_Bulge', '31_Skel', '31_Bulge','26_Skel', '26_Bulge'}, {'31_0_Skel', '31_0_Bulge','31_50_Skel', '31_50_Bulge'}};

skel = {{'23','31', '26'},{'31_0', '31_50'}};
SF = {'Skel', 'Bulge'};

imScale = .2;
imTrans = .2;
frames = 300; %Number of frames

%rF = randperm(frames,20);


for ee = 1:length(exp)
	allData = {};
	n=1;

	for sk = 1:length(skel{ee})
		for sf = 1:length(SF)
            stimActs = [];
            sizeActs = [];
			
            st = 1;
			for fn = 1:frames
				%Load original image
				ogIM = imread(['Frames/Figure_', skel{ee}{sk},'_',SF{sf}, '_', int2str(fn), '.jpg']);
                sizeIM = zeros(round(size(ogIM,1)*(imScale+1)), round(size(ogIM,1)*(imScale+1)),3,'uint8');
                sizeIM(:,:,1) = 119;
                sizeIM(:,:,2) = 119;
                sizeIM(:,:,3) = 119;
                %Overlay OG image on blank in top left corner
                %This reduces the size and shifts it by imScale %
                sizeIM(1:size(ogIM,1), 1:size(ogIM,1), :) = ogIM;
                
                %padSize = round(((size(ogIM,1)*(imScale+1))- size(ogIM,1))/2,0);

				%Resize images by 25%
				%sizeIM = padarray(ogIM, [0, padSize*2], 119); %Pad with 119 (gray) by % size reduction
				
				%Load diff SF image
				if strcmp(SF{sf},'Skel')
					sfIM = imread(['Frames/Figure_', skel{ee}{sk},'_Bulge_', int2str(fn), '.jpg']);
				else
					sfIM = imread(['Frames/Figure_', skel{ee}{sk},'_Skel_', int2str(fn), '.jpg']);
				end
				
				%Resize to GBJ input size
				ogIM = imresize(ogIM, [256,256]);
				sizeIM = imresize(sizeIM, [256,256]);
				sfIM = imresize(sfIM, [256,256]);

				%Extract Gabor Magnitudes
				ogGBJ = GWTWgrid_Simple(ogIM, 0, 1);
				sizeGBJ = GWTWgrid_Simple(sizeIM, 0, 1);
				sfGBJ = GWTWgrid_Simple(sfIM, 0, 1);

				%Vectorize and transpose
				ogGBJ = ogGBJ(:)';
				sizeGBJ = sizeGBJ(:)';
				sfGBJ = sfGBJ(:)';
              
                %Save to matrix
				stimActs(st,:) = ogGBJ;
				sizeActs(st,:) = sizeGBJ;
                
				%Calculate diff
				sizeDiff = norm(ogGBJ - sizeGBJ);
				sfDiff = norm(ogGBJ - sfGBJ);



				%Save results
				allData{n,1} = skel{ee}{sk};
                allData{n,2} = SF{sf};
                allData{n,3} = fn;
                allData{n,4} = sfDiff;
                allData{n,5} = sizeDiff;

                n= n +1;
                st = st+1;

			end
			%Save GBJ acts as file
			save(['GBJ_Acts/Figure_', skel{ee}{sk},'_',SF{sf}, '_GBJActs'], 'stimActs');
			save(['GBJ_Acts/Figure_', skel{ee}{sk},'_',SF{sf}, '_GBJActs_Size', int2str(imScale*100)], 'sizeActs');
                
        end
        
	end
	allData = cell2table(allData, 'VariableNames', {'Skel', 'SF', 'Frame', 'sfDiff', 'sizeDiff'});
	writetable(allData, ['Results/LiMA_', exp{ee}, '_GBJ_diffs_',  int2str(imScale*100), '.csv']);
	

end

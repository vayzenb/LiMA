%% 
%Makes frames textured
%%
clear all;
tex = im2double(imread('checker.jpg'));

    imFiles = dir('Frames\*.jpg');
    for kk = 1:length(imFiles)
        % Read in original image
        IM = imread(['Frames\', imFiles(kk).name]);
        %convert to matrix
        IM = im2double(IM);
        
        %make object in image a binary mask
        BW = imbinarize(rgb2gray(IM));
        BW = imcomplement(BW); %invert it
        BW2 = imfill(BW,'holes'); %Fill in the holes
        BW3 = imcomplement(BW2); %invert it back
        
        tex2 = tex .* BW2; %mask the texture image
        
        IM2 = IM .* BW3; % Mask original image
        IM3 = IM2 + tex2; %Add texture to the original image by adding it
        

        imwrite(IM3, ['Frames\', imFiles(kk).name(1:end-4), '_tex.jpg']);
    end

    
    

    %
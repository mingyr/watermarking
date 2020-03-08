clear;
clc;

dirs = dir('/scratch/flowers/flowers');
dirs = dirs(~ismember({dirs.name}, {'.', '..'}));

W = 320;
H = 240;

image_paths = {};
count = 0;

for i = length(dirs) : length(dirs)
    d = dirs(i);
    ptn = fullfile(d.folder, d.name, '*.jpg');
    files = dir(ptn);
    for j = 1 : length(files)
        file = files(j);
        image_path = fullfile(file.folder, file.name);
        image = imread(image_path, 'jpg');
        [h, w, c] = size(image);
        if w < 320 || h < 240
            continue;
        end
        
        count = count + 1;
        image_paths{count} = image_path;
        
    end
    break;
end

save('image_paths.mat', 'image_paths');

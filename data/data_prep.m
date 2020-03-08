clear;
clc;

dirs = dir('/scratch/flowers/flowers');
dirs = dirs(~ismember({dirs.name}, {'.', '..'}));

train_images = uint8([]);
train_count = 0;

test_images = uint8([]);
test_count = 0;

W = 320;
H = 240;

for i = 1 : (length(dirs) - 1)
    d = dirs(i);
    ptn = fullfile(d.folder, d.name, '*.jpg');
    files = dir(ptn);
    for j = 1 : length(files)
        file = files(j);
        image = imread(fullfile(file.folder, file.name), 'jpg');
        [h, w, c] = size(image);
        if w < 320 || h < 240
            continue;
        end
        
        w_offset = floor((w - W) / 2);
        h_offset = floor((h - H) / 2);
        
        image = image((1 + h_offset):(1 + h_offset + H - 1), ...
                      (1 + w_offset):(1 + w_offset + W - 1), :);
        
        train_count = train_count + 1;
        train_images(:, :, :, train_count) = uint8(image);
    end
end

for i = length(dirs) : length(dirs)
    d = dirs(i);
    ptn = fullfile(d.folder, d.name, '*.jpg');
    files = dir(ptn);
    for j = 1 : length(files)
        file = files(j);
        image = imread(fullfile(file.folder, file.name), 'jpg');
        [h, w, c] = size(image);
        if w < 320 || h < 240
            continue;
        end
        
        w_offset = floor((w - W) / 2);
        h_offset = floor((h - H) / 2);
        
        image = image((1 + h_offset):(1 + h_offset + H - 1), ...
                      (1 + w_offset):(1 + w_offset + W - 1), :);

        test_count = test_count + 1;
        test_images(:, :, :, test_count) = uint8(image);
        
    end
    break;
end

save('image_data.mat', 'train_images', 'test_images', '-v7.3');

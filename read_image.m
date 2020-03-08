function image = read_image(image_path)
    W = 320;
    H = 240;

    image = imread(image_path, 'jpg');

    [h, w, c] = size(image);
    if w < 320 || h < 240
        image = -1;
    else
        w_offset = floor((w - W) / 2);
        h_offset = floor((h - H) / 2);
        
        image = image((1 + h_offset):(1 + h_offset + H - 1), ...
                  (1 + w_offset):(1 + w_offset + W - 1), :);
    end
end


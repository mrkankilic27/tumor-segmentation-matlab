function masks = enhancedSegmentation(image, bboxes)
    masks = {};
    
    % Görüntüyü gri tonlamalıya çevir (RGB ise)
    if size(image,3) == 3
        grayImage = rgb2gray(image);
    else
        grayImage = image; % Zaten gri tonlamalı
    end

    for i = 1:size(bboxes,1)
        box = round(bboxes(i,:));
        x = max(box(1), 1); % X koordinatı en az 1 olmalı
        y = max(box(2), 1); % Y koordinatı en az 1 olmalı
        w = box(3);
        h = box(4);
        
        % Bounding box'ın görüntü sınırlarını aşmamasını sağla
        w = min(w, size(grayImage,2) - x + 1);
        h = min(h, size(grayImage,1) - y + 1);
        
        % Geçersiz boyut kontrolü
        if w <= 0 || h <= 0
            continue;
        end

        % ROI'yi al
        subImage = imcrop(grayImage, [x y w h]);

        % Adaptif eşikleme
        bw = imbinarize(subImage, 'adaptive', 'Sensitivity', 0.7);

        % Gürültü temizleme
        bw = bwareaopen(bw, 30);
        bw = imfill(bw, 'holes');

        % Morfolojik işlemler
        se = strel('disk', 2);
        bw = imopen(bw, se);
        bw = imclose(bw, se);

        % Maske oluştur (boyutları tam olarak eşleştir)
        mask = false(size(grayImage));
        try
            mask(y:y+h-1, x:x+w-1) = bw(1:h, 1:w);
        catch ME
            % Boyut uyumsuzluğu durumunda otomatik ayarlama
            actualH = min(size(bw,1), h);
            actualW = min(size(bw,2), w);
            mask(y:y+actualH-1, x:x+actualW-1) = bw(1:actualH, 1:actualW);
        end
        
        masks{end+1} = mask;
    end
end
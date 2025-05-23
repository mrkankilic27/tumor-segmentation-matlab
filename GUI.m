function tumorDetectionGUI()
    fig = figure('Name', 'Tümör Tespit ve Segmentasyon Sistemi', ...
                'Position', [100 100 1200 800], ...
                'NumberTitle', 'off', ...
                'MenuBar', 'none', ...
                'ToolBar', 'none', ...
                'Color', [0.94 0.94 0.94]);

    % GUI bileşenleri
    uicontrol('Style', 'text', ...
              'String', 'TÜMÖR TESPİT ve SEGMENTASYON SİSTEMİ', ...
              'Position', [350 750 500 30], ...
              'FontSize', 16, ...
              'FontWeight', 'bold', ...
              'BackgroundColor', [0.94 0.94 0.94]);

    uicontrol('Style', 'pushbutton', ...
              'String', 'Veri Yükle', ...
              'Position', [50 700 150 40], ...
              'FontSize', 12, ...
              'Callback', @loadData);

    uicontrol('Style', 'pushbutton', ...
              'String', 'Model Yükle', ...
              'Position', [220 700 150 40], ...
              'FontSize', 12, ...
              'Callback', @loadModel);

    uicontrol('Style', 'pushbutton', ...
              'String', 'Resim Seç', ...
              'Position', [390 700 150 40], ...
              'FontSize', 12, ...
              'Callback', @selectImage);

    uicontrol('Style', 'pushbutton', ...
              'String', 'Analiz Yap', ...
              'Position', [560 700 150 40], ...
              'FontSize', 12, ...
              'Callback', @runAnalysis);

    % Görüntüleme alanları
    ax1 = axes('Parent', fig, 'Units', 'pixels', 'Position', [50 350 500 300]);
    ax2 = axes('Parent', fig, 'Units', 'pixels', 'Position', [600 350 500 300]);

    infoPanel = uicontrol('Style', 'text', ...
                         'String', 'Sistem hazır...', ...
                         'Position', [50 50 1100 250], ...
                         'FontSize', 12, ...
                         'HorizontalAlignment', 'left', ...
                         'BackgroundColor', 'white');

    % Global değişkenler
    detector = [];
    groundTruthData = [];
    groundTruthFilenames = {};
    selectedImage = [];
    currentResults = struct('Image', [], 'YOLO_Boxes', [], 'YOLO_Scores', [], ...
                           'Segmentation_Masks', [], 'TrueBoxes', []);

    % Veri yükleme fonksiyonu
    function loadData(~, ~)
        try
            [file, path] = uigetfile('*.mat', 'YOLOv2 Dataset Dosyasını Seç');
            if isequal(file, 0); return; end
            
            data = load(fullfile(path, file));
            if ~isfield(data, 'T'); error('T tablosu bulunamadı'); end
            
            % Görüntü yollarını güncelle
            tumorDir = fullfile(pwd, 'tumor');
            for i = 1:height(data.T)
                [~, name, ext] = fileparts(data.T.imageFilename{i});
                data.T.imageFilename{i} = fullfile(tumorDir, [name ext]);
            end
            
            % Veri setini oluştur (RGB formatında okuma garantisi)
            imds = imageDatastore(data.T.imageFilename, 'ReadFcn', @ensureRGB);
            blds = boxLabelDatastore(data.T(:,2));
            ds = combine(imds, blds);
            
            % Tüm veriyi oku
            groundTruthData = readall(ds);
            groundTruthFilenames = imds.Files;
            
            set(infoPanel, 'String', sprintf('Veri başarıyla yüklendi! Toplam %d görüntü.', length(groundTruthData)));
        catch e
            set(infoPanel, 'String', sprintf('HATA: %s', e.message));
        end
    end

    % Model yükleme fonksiyonu
    function loadModel(~, ~)
        try
            [file, path] = uigetfile('*.mat', 'YOLOv2 Model Dosyasını Seç');
            if isequal(file, 0); return; end
            
            loaded = load(fullfile(path, file));
            if ~isfield(loaded, 'detector'); error('detector bulunamadı'); end
            
            detector = loaded.detector;
            set(infoPanel, 'String', sprintf('Model başarıyla yüklendi: %s', file));
        catch e
            set(infoPanel, 'String', sprintf('HATA: %s', e.message));
        end
    end

    % Resim seçme fonksiyonu
    function selectImage(~, ~)
        [file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.tif', 'Image Files'}, 'Resim Seç');
        if isequal(file, 0); return; end
        
        try
            % Resmi oku ve RGB formatına dönüştür
            selectedImage = ensureRGB(imread(fullfile(path, file)));
            currentResults.Image = selectedImage;
            currentResults.YOLO_Boxes = [];
            currentResults.YOLO_Scores = [];
            currentResults.Segmentation_Masks = [];
            
            % Ground truth eşleştirme
            currentResults.TrueBoxes = [];
            if ~isempty(groundTruthFilenames)
                [~, currentName, currentExt] = fileparts(file);
                currentFullName = lower([currentName currentExt]);
                
                for i = 1:length(groundTruthFilenames)
                    [~, gtName, gtExt] = fileparts(groundTruthFilenames{i});
                    gtFullName = lower([gtName gtExt]);
                    
                    if strcmp(currentFullName, gtFullName)
                        currentResults.TrueBoxes = groundTruthData{i,2};
                        break;
                    end
                end
            end
            
            % Görüntüyü göster
            axes(ax1); imshow(selectedImage); title('Seçilen Görüntü');
            axes(ax2); cla; title('Segmentasyon Sonuçları');
            
            if ~isempty(currentResults.TrueBoxes)
                set(infoPanel, 'String', sprintf('Resim yüklendi: %s\nGround truth bulundu!', file));
            else
                set(infoPanel, 'String', sprintf('Resim yüklendi: %s\nGround truth bulunamadı.', file));
            end
        catch e
            set(infoPanel, 'String', sprintf('HATA: %s', e.message));
        end
    end

    % Analiz fonksiyonu
    function runAnalysis(~, ~)
        if isempty(detector)
            set(infoPanel, 'String', 'Önce model yüklemelisiniz.'); 
            return;
        end
        if isempty(selectedImage)
            set(infoPanel, 'String', 'Önce bir resim seçmelisiniz.'); 
            return;
        end
        
        try
            % YOLOv2 tespiti (RGB format garantili)
            [bboxes, scores] = detect(detector, selectedImage, 'Threshold', 0.5);
            
            % Segmentasyon (orijinal resim üzerinde)
            masks = enhancedSegmentation(selectedImage, bboxes);
            
            % Sonuçları kaydet
            currentResults.YOLO_Boxes = bboxes;
            currentResults.YOLO_Scores = scores;
            currentResults.Segmentation_Masks = masks;
            
            % Görüntüle
            updateDisplay();
            set(infoPanel, 'String', 'Tespit ve segmentasyon tamamlandı.');
        catch e
            set(infoPanel, 'String', sprintf('HATA: %s', e.message));
        end
    end

    % Görüntüleme fonksiyonu
    function updateDisplay()
        % Tespit sonuçları
        axes(ax1);
        imshow(currentResults.Image); hold on;
        
        % Ground truth (yeşil)
        if ~isempty(currentResults.TrueBoxes)
            for j = 1:size(currentResults.TrueBoxes, 1)
                rectangle('Position', currentResults.TrueBoxes(j,:), ...
                         'EdgeColor', 'g', 'LineWidth', 2);
            end
        end
        
        % YOLO tespitleri (kırmızı)
        if ~isempty(currentResults.YOLO_Boxes)
            for j = 1:size(currentResults.YOLO_Boxes, 1)
                rectangle('Position', currentResults.YOLO_Boxes(j,:), ...
                         'EdgeColor', 'r', 'LineWidth', 2);
                text(currentResults.YOLO_Boxes(j,1), currentResults.YOLO_Boxes(j,2)-10, ...
                    sprintf('%.2f', currentResults.YOLO_Scores(j)), ...
                    'Color', 'w', 'BackgroundColor', 'r', 'FontSize', 8);
            end
        end
        hold off;
        title('Tespit Sonuçları (Yeşil: GT, Kırmızı: Tahmin)');
        
        % Segmentasyon sonuçları
        axes(ax2);
        imshow(currentResults.Image); hold on;
        
        if ~isempty(currentResults.Segmentation_Masks)
            for j = 1:length(currentResults.Segmentation_Masks)
                if ~isempty(currentResults.Segmentation_Masks{j})
                    B = bwboundaries(currentResults.Segmentation_Masks{j});
                    for k = 1:length(B)
                        boundary = B{k};
                        plot(boundary(:,2), boundary(:,1), 'c', 'LineWidth', 1.5);
                    end
                end
            end
        end
        hold off;
        title('Segmentasyon Sonuçları');
    end

    % Yardımcı fonksiyon: Tüm görüntüleri RGB formatına dönüştür
    function img = ensureRGB(img)
        if ischar(img) || isstring(img)
            img = imread(img);
        end
        
        % Kanal sayısını kontrol et ve dönüştür
        if size(img, 3) == 1
            img = cat(3, img, img, img); % Gri seviyeden RGB'ye
        elseif size(img, 3) == 4
            img = img(:,:,1:3); % Alpha kanalını kaldır
        end
        
        % Görüntüyü uint8 formatına dönüştür (gerekiyorsa)
        if ~isa(img, 'uint8')
            img = im2uint8(img);
        end
    end
end

% Geliştirilmiş segmentasyon fonksiyonu
function masks = enhancedSegmentation(image, bboxes)
    masks = {};
    grayImage = rgb2gray(image);

    for i = 1:size(bboxes,1)
        box = round(bboxes(i,:));
        x = max(box(1),1);
        y = max(box(2),1);
        w = box(3);
        h = box(4);

        % ROI'yi al
        subImage = imcrop(grayImage, [x y w h]);

        % Basit eşikleme
        level = graythresh(subImage);
        bw = imbinarize(subImage, level);

        % Gürültü temizleme ve maske üretimi
        bw = imfill(bw, 'holes');
        bw = bwareaopen(bw, 30);

        % Hedef maske boyutunu sınırla
        targetH = min(size(bw,1), size(grayImage,1)-y+1);
        targetW = min(size(bw,2), size(grayImage,2)-x+1);

        % Maske oluştur
        mask = false(size(grayImage));
        mask(y:y+targetH-1, x:x+targetW-1) = bw(1:targetH, 1:targetW);

        masks{end+1} = mask;
    end
end
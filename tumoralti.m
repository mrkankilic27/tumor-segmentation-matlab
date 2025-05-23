clc; clear; close all; warning off;

%% 1. Veri Yükleme ve Hazırlık
disp("1. Veri yükleniyor ve hazırlanıyor...");
load('YOLOv2_dataset.mat'); % 'T' tablosu yükle

% Görüntü yollarını düzelt
goruntuKlasoru = fullfile(pwd, 'tumor');
if ~isfolder(goruntuKlasoru)
    error('Tumor klasörü bulunamadı: %s', goruntuKlasoru);
end

for i = 1:height(T)
    [~, name, ext] = fileparts(T.imageFilename{i});
    T.imageFilename{i} = fullfile(goruntuKlasoru, [name, ext]);
    
    if ~isfile(T.imageFilename{i})
        error('Dosya bulunamadı: %s', T.imageFilename{i});
    end
end

% Veri depolarını oluştur
imds = imageDatastore(T.imageFilename, 'ReadFcn', @readAndConvertRGB);
blds = boxLabelDatastore(T(:,2));
ds = combine(imds, blds);

%% 2. Veri Bölme
disp("2. Veri seti bölünüyor (70-15-15)...");
numData = height(T);
idx = randperm(numData);

numTrain = round(0.7 * numData);
numVal = round(0.15 * numData);
numTest = numData - numTrain - numVal;

dsTrain = subset(ds, idx(1:numTrain));
dsVal = subset(ds, idx(numTrain+1:numTrain+numVal));
dsTest = subset(ds, idx(numTrain+numVal+1:end));

%% 3. Gelişmiş Veri Artırma
disp("3. Gelişmiş veri artırma işlemleri yapılıyor...");
augmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [-10 10], ...
    'RandScale', [0.7 1.3], ...
    'RandXShear', [-20 20], ...
    'RandYShear', [-20 20], ...
    'RandXTranslation', [-30 30], ...
    'RandYTranslation', [-30 30], ...
    'FillValue', 0);

augmentedTrainingData = transform(dsTrain, @(data)augmentData(data, augmenter));

%% 4. YOLOv2 Model Oluşturma
disp("4. YOLOv2 ağı oluşturuluyor...");
inputSize = [224 224 3];
numClasses = 1;
classNames = {'tumor'};

baseNetwork = resnet50();
featureLayer = 'activation_40_relu';

% Anchor box'ları tahmin et
preprocessedData = transform(dsTrain, @(data)preprocessData(data, inputSize));
anchorBoxes = estimateAnchorBoxes(preprocessedData, 3); % 3 anchor box kullan

% YOLOv2 ağı oluştur
lgraph = yolov2Layers(inputSize, numClasses, anchorBoxes, baseNetwork, featureLayer);

%% 5. Gelişmiş Model Eğitimi
disp("5. Gelişmiş eğitim başlatılıyor...");
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.1, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 8, ...
    'VerboseFrequency', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', dsVal, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'CheckpointPath', tempdir);

[detector, info] = trainYOLOv2ObjectDetector(augmentedTrainingData, lgraph, options);

%% 6. Model Değerlendirme
disp("6. YOLOv2 test ediliyor...");
detectionResults = detect(detector, dsTest, 'MiniBatchSize', 4, 'Threshold', 0.5);
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults, dsTest);

fprintf('\n--- YOLOv2 Değerlendirme ---\n');
fprintf('Average Precision (AP): %.4f\n', ap);
fprintf('Recall: %.2f%%\n', recall(end)*100);
fprintf('Precision: %.2f%%\n', precision(end)*100);

%% 7. Gelişmiş Segmentasyon Fonksiyonu
disp("7. Gelişmiş segmentasyon işlemleri başlatılıyor...");

%% 8. Hibrit Tespit ve Segmentasyon
disp("8. Hibrit analiz çalıştırılıyor...");
testData = readall(dsTest);
results = struct();

for i = 1:min(5, size(testData,1)) % İlk 5 örnek için
    I = testData{i,1};
    trueBoxes = testData{i,2};
    
    % YOLOv2 ile tespit (düşük threshold ile)
    [bboxes, scores] = detect(detector, I, 'Threshold', 0.1);
    
    % Gelişmiş segmentasyon
    masks = enhancedSegmentation(I, bboxes);
    
    % Sonuçları kaydet
    results(i).Image = I;
    results(i).TrueBoxes = trueBoxes;
    results(i).YOLO_Boxes = bboxes;
    results(i).YOLO_Scores = scores;
    results(i).Segmentation_Masks = masks;
end

%% 9. Gelişmiş Görselleştirme
disp("9. Sonuçlar görselleştiriliyor...");
num_samples = numel(results);
fig_height = max(400, 300 * num_samples);

figure('Name','Tümör Tespit ve Segmentasyon','Position',[100 100 1400 fig_height], 'Color', 'w');

for i = 1:num_samples
    % Tespit Görseli (Sol)
    subplot(num_samples, 2, (i-1)*2+1);
    imshow(results(i).Image);
    hold on;
    
    % Ground Truth (Yeşil)
    for j = 1:size(results(i).TrueBoxes,1)
        rectangle('Position', results(i).TrueBoxes(j,:), ...
                'EdgeColor', [0 0.8 0], 'LineWidth', 2.5, 'LineStyle', '-');
        text(results(i).TrueBoxes(j,1), results(i).TrueBoxes(j,2)-15, 'GT', ...
            'Color', 'w', 'BackgroundColor', [0 0.8 0], 'FontSize', 10);
    end
    
    % YOLO Tespitleri (Kırmızı)
    if ~isempty(results(i).YOLO_Boxes)
        for j = 1:size(results(i).YOLO_Boxes,1)
            rectangle('Position', results(i).YOLO_Boxes(j,:), ...
                    'EdgeColor', [0.9 0 0], 'LineWidth', 2.5);
            text(results(i).YOLO_Boxes(j,1), results(i).YOLO_Boxes(j,2)-5, ...
                sprintf('%.2f', results(i).YOLO_Scores(j)), ...
                'Color', 'w', 'BackgroundColor', [0.7 0 0], 'FontSize', 10);
        end
    else
        text(20, 30, 'YOLO: Tespit Yok', ...
            'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');
    end
    title(sprintf('Örnek %d - Tespitler', i), 'FontSize', 12);
    hold off;
    
    % Segmentasyon Görseli (Sağ)
    subplot(num_samples, 2, (i-1)*2+2);
    imshow(results(i).Image);
    hold on;
    
    % Segmentasyon Maskeleri
    if ~isempty(results(i).Segmentation_Masks)
        colors = lines(length(results(i).Segmentation_Masks));
        for j = 1:min(3, length(results(i).Segmentation_Masks))
            contour(results(i).Segmentation_Masks{j}, 'Color', colors(j,:), 'LineWidth', 2.2);
            text(15, 15+30*(j-1), sprintf('Maske %d', j), ...
                'Color', colors(j,:), 'FontSize', 11, 'FontWeight', 'bold');
        end
    else
        text(20, 30, 'Segmentasyon: Maske Yok', ...
            'Color', 'c', 'FontSize', 12, 'FontWeight', 'bold');
    end
    title(sprintf('Örnek %d - Segmentasyon', i), 'FontSize', 12);
    hold off;
end

%% 10. Sonuçları Kaydet ve Raporla
disp("10. Sonuçlar kaydediliyor...");

% Mevcut klasöre kaydetmek için
results_dir = fullfile(pwd, 'tumor_results');

% Klasör oluşturma işlemi (güvenli versiyon)
try
    if ~isfolder(results_dir)
        mkdir(results_dir);
    end
    fprintf('Sonuçlar buraya kaydedilecek: %s\n', results_dir);
catch e
    % Hata durumunda alternatif yol
    warning('Birincil klasör oluşturulamadı. Temp dizin kullanılıyor...');
    results_dir = fullfile(tempdir, 'tumor_results');
    mkdir(results_dir);
end

% Sonuçları kaydet
output_file = fullfile(results_dir, ['tumor_results_' datestr(now,'yyyymmdd_HHMM') '.mat']);
save(output_file, 'results', 'detector', 'info', '-v7.3');

% Görselleri kaydet
for i = 1:numel(results)
    fig = figure('Visible','off');
    imshow(results(i).Image);
    hold on;
    
    if ~isempty(results(i).YOLO_Boxes)
        for j = 1:size(results(i).YOLO_Boxes,1)
            rectangle('Position', results(i).YOLO_Boxes(j,:), ...
                    'EdgeColor', [0.9 0 0], 'LineWidth', 2);
        end
    end
    
    if ~isempty(results(i).Segmentation_Masks)
        for j = 1:min(3, length(results(i).Segmentation_Masks))
            contour(results(i).Segmentation_Masks{j}, 'Color', 'c', 'LineWidth', 2);
        end
    end
    hold off;
    
    exportgraphics(fig, fullfile(results_dir, sprintf('result_%d.png',i)), 'Resolution',300);
    close(fig);
end

fprintf('✅ Analiz tamamlandı! Sonuçlar kaydedildi:\n- %s\n- %s klasöründeki görseller\n', output_file, results_dir);

%% Yardımcı Fonksiyonlar
function I = readAndConvertRGB(filename)
    I = imread(filename);
    if size(I, 3) == 1
        I = repmat(I, 1, 1, 3);
    end
    I = im2single(I);
end

function dataOut = augmentData(data, augmenter)
    I = data{1};
    bbox = data{2};
    label = data{3};

    if size(I, 3) == 1
        I = repmat(I, 1, 1, 3);
    end

    if rand > 0.5
        I = fliplr(I);
        bbox(:,1) = size(I,2) - bbox(:,1) - bbox(:,3) + 1;
    end

    try
        augmented = augment(augmenter, I, bbox);
        I = augmented{1};
        bbox = augmented{2};
    catch
        warning('Augmentasyon uygulanamadı, orijinal görüntü kullanılıyor');
    end

    dataOut = {I, bbox, label};
end

function data = preprocessData(data, targetSize)
    I = data{1};
    bbox = data{2};
    label = data{3};

    if size(I, 3) == 1
        I = repmat(I, 1, 1, 3);
    end

    originalSize = size(I);
    I = im2single(imresize(I, targetSize(1:2)));
    scale = targetSize(1:2) ./ originalSize(1:2);
    bbox = bbox .* [scale(2) scale(1) scale(2) scale(1)];

    data = {I, bbox, label};
end

function masks = enhancedSegmentation(I, bboxes)
    masks = {};
    
    % Görüntü ön işleme
    if size(I,3) == 3
        grayImg = rgb2gray(I);
    else
        grayImg = I;
    end
    
    % Kontrast sınırla (aşırı artırma yapma)
    grayImg = imadjust(grayImg, stretchlim(grayImg, [0.01 0.99]), []);
    
    % Gürültü azaltma (daha hafif bir filtre)
    grayImg = imgaussfilt(grayImg, 1);
    
    for i = 1:size(bboxes,1)
        try
            % ROI çıkarma ve boyut kontrolü
            x = floor(bboxes(i,1));
            y = floor(bboxes(i,2));
            w = ceil(bboxes(i,3));
            h = ceil(bboxes(i,4));
            
            % Minimum ROI boyutu kısıtlaması
            if w < 20 || h < 20  % Çok küçük bölgeleri atla
                continue;
            end
            
            % Sınır kontrolü
            x = max(1, x); y = max(1, y);
            w = min(size(grayImg,2)-x, w);
            h = min(size(grayImg,1)-y, h);
            
            roi = grayImg(y:y+h, x:x+w);
            
            % 1. Adaptif eşikleme (daha hassas parametreler)
            bw1 = imbinarize(roi, 'adaptive', 'Sensitivity', 0.4);
            
            % 2. Active Contour (daha kontrollü)
            initialMask = false(size(roi));
            center = round(size(roi)/2);
            radius = round(min(size(roi))/4); % Daha küçük başlangıç alanı
            [xx,yy] = meshgrid(1:size(roi,2), 1:size(roi,1));
            initialMask = (xx-center(2)).^2 + (yy-center(1)).^2 <= radius^2;
            
            % Daha kontrollü active contour parametreleri
            bw2 = activecontour(roi, initialMask, 100, 'Chan-Vese', ...
                               'SmoothFactor', 1.5, 'ContractionBias', -0.2);
            
            % 3. Morfolojik temizleme
            combinedMask = bw1 & bw2; % İkisinin de kesişimini al
            combinedMask = bwareaopen(combinedMask, 50); % Küçük nesneleri sil
            combinedMask = imfill(combinedMask, 'holes');
            
            % Kontur düzgünlüğü kontrolü
            stats = regionprops(combinedMask, 'Solidity', 'Area');
            if ~isempty(stats) && stats.Solidity < 0.7 % Çok düzensiz şekilleri ele
                continue;
            end
            
            % Ana maskeye ekle
            fullMask = false(size(grayImg));
            fullMask(y:y+h, x:x+w) = combinedMask;
            
            % Örtüşen maskeleri kontrol et
            overlap = false;
            for m = 1:length(masks)
                if any(any(fullMask & masks{m}))
                    overlap = true;
                    break;
                end
            end
            
            if ~overlap
                masks{end+1} = fullMask;
            end
            
        catch e
            warning('Segmentasyon hatası: %s', e.message);
            continue;
        end
    end
end
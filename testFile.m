clc;
clear;
close all;

opts = detectImportOptions('referens.csv');
data = readtable('referens.csv');  
data2 = readtable('test.csv');  

varNames = {'Z_MEAN', 'Z_STD', 'SLOPE', 'URBAN', 'WATER', 'DJUR', 'NRSV', 'KRSV', 'NPARK', 'NOMR', 'ROAD', 'EL', 'WIND', 'TURBINE'};

data = data{:, 2:15};
data2 = data2{:, 2:15};

allData = [data; data2];  
minValues = min(allData);
maxValues = max(allData);
normalizedData = (data - minValues) ./ (maxValues - minValues);
normalizedData2 = (data2 - minValues) ./ (maxValues - minValues); 


figure;


parallelcoords(normalizedData2, 'LineWidth', 0.5, 'Labels', varNames, 'Color', [1 0 0 0.1]); 
hold on;


parallelcoords(normalizedData, 'LineWidth', 0.5, 'Labels', varNames, 'Color', [0.5, 0.75, 1, 0.1]);  


plot(mean(normalizedData), '-o', 'LineWidth', 1.5, 'Color', [0 0 1]); 
plot(mean(normalizedData2), '-o', 'LineWidth', 1.5, 'Color', [1 0 0]); 

title('Parallel Coordinate Plot with Highlighted Mean (First 14 Variables)');
xlabel('Variables');
ylabel('Normalized Values');
grid on;

legend({'Dataset 1', 'Dataset 2', 'Mean (Dataset 1)', 'Mean (Dataset 2)'}, 'Location', 'northeastoutside');

axis tight;
%%Mohammed ändring

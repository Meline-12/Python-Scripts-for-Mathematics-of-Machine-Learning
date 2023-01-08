% MATHEMATIK DES MASCHINELLEN LERNENS
%===========================================
% Kapitel 3: Lineare Klassifikationsmethoden
% Abschnitt 3.4: Weiche SVM-Regel
%
% MATLAB-Skript zum Nachvollziehen des 
% MNIST-Beispiels zur weichen SVM-Regel

% Alles loeschen und schliessen
clear all; close all; clc;

%% Laden der Daten:
load('data_MNIST_78.mat'); 
Y = (Y == 7) - (Y == 8);
y = Y';
m = length(y); % Anzahl Daten
d = size(X,1); % Anzahl Merkmale (= Dimension des Merkmalsraums)

%% (1) Weiche SVM-Regel
%----------------------------

lam = 1/m; % beide Terme gleichgewichtet

% Die Lösung bestimmen
SVM = fitcsvm(X',y, 'ClassNames',[-1,1],'BoxConstraint', 1/(2*m*lam));

% HINWEIS: 'BoxConstraint' entspricht 1/(2*m*lambda)

% Fehlklassifikationsrate bestimmen
label = predict(SVM,X');
mean(label ~= y')

% Support-Vektoren finden
ind = find(SVM.IsSupportVector);

% Gewichtsvektor berechnen
w_S = X(:, ind) * (SVM.Alpha .* y(ind)');
SVM.Alpha

% Darstellen des Gewichtsvektors
figure()
plot(w_S,'o','LineWidth',2)
grid on;
xlabel('Komponente k')
ylabel('Gewicht w_k')
title('Gewichtsvektor w_S für \lambda = 1/m')
set(gca,'FontSize',16)

% Fehlklassifikationen finden
ind_wrong = find( y .* (w_S' * X + SVM.Bias) < 0);
SVM.Bias

% Zeichnen der fehlklassifizierten Bilder
figure();
for i=1:length(ind_wrong),
    subplot(5,5,i);
    x = X(:,ind_wrong(i));
    imshow(reshape(x, 28,28)',[0,1]);
end

%% Einfluss von Lambda uzntersuchen
rate=zeros(20,1);
tic;
for j = 1:20,
    lam = 2^(10-j+1); % Wert von Lambda
    
    % Trainieren fuer dieses Lambda
    SVM = fitcsvm(X',y, 'ClassNames',[-1,1],'BoxConstraint', 1/(2*m*lam));
    
    % Vorhergesagt Labels an den Trainingsdaten
    label = predict(SVM,X');
    
    % Fehlklassifikationsrate
    rate(j) = mean(label ~= y');
end
toc;

%% Plotten
figure()
semilogx(2.^(11 - (1:20)), rate,'o-','Linewidth',2)
grid on
ylim([0,0.5])
xlim([1e-3,2e3])
xlabel('\lambda')
ylabel('Fehlklassifikation')
title('Fehlklassifikation in den Trainingsdaten')
set(gca,'FontSize',16)
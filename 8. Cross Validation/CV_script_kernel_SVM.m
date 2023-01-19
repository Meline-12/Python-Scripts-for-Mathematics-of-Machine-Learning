%% Cross Validation

% Alles loeschen und schliessen
clear all; close all; clc;

% Daten laden
load('data_KSVM');

% Daten zufällig anordnen
m = length(y);
% permutation = randperm(m);
% 
% y = y(permutation);
% x = x(:, permutation);

% Zeichnen der Trainingsdaten:
figure(1)

indp = find(y>0); % Daten mit +1 
indm = find(y<0); % Daten mit -1 

plot(x(1,indp),x(2,indp),'ob','Linewidth',2); hold on;
plot(x(1,indm),x(2,indm),'+r','Linewidth',2); hold on;
grid on;
axis tight;
xlabel('x_1')
ylabel('x_2')
set(gca,'fontsize',14)

%% a) und b) Cross-Validation

% Lambda fuer weiche Kern-SVM-Regel:
lam = 0.5/m;

% Anzahl Blöcke bzw. folds
K = 10;
batch_size = m/K;

% Skalierungsparameter:
kappas = 2.^(-2:0.25:3);

% Schleife über kappa-Werte
CV_error = zeros(1,length(kappas));
tic;
for j = 1:length(kappas),
    kappa = kappas(j);
    
    % Cross-Validation-Schleife über Blöcke
    for k = 1:K,
        indizes_test = (k-1)*batch_size + (1:batch_size);
        indizes_train = setdiff(1:m,indizes_test);
        
        % Trainigsdaten
        x_train = x(:, indizes_train);
        y_train = y(indizes_train);
        
        % Lernen aus Trainingsdaten
        KSVM = fitcsvm(x_train',y_train, 'ClassNames',[-1,1],...
            'KernelFunction','rbf', 'KernelScale', 1/sqrt(kappa), ...
            'BoxConstraint', 0.5/m/lam);

        % Testsdaten
        x_test = x(:, indizes_test);
        y_test = y(indizes_test);
        
        % CV-Fehler aufdatieren für Verlust
        h_test = predict(KSVM, x_test');
        CV_error(j) = CV_error(j) + mean(y_test' ~= h_test);
     
    end
    
    CV_error(j) = CV_error(j)/K;
  
end
toc;

% Zeichnen
figure(2)
plot(kappas, CV_error,'o-','linewidth',2);
grid on;
xlabel('\kappa')
ylabel('CV-Error')
title('Kreuzvalidierung mit 01-Verlust')
set(gca,'FontSize',16)


%% c) Beste und schlechteste Hypothese neu lernen

[CV_min, j_min] = min(CV_error)
[CV_max, j_max] = max(CV_error)
kappa_best = kappas(j_min);
kappa_worst = kappas(j_max);

% Lernen aus ALLEN Trainingsdaten
KSVM_best = fitcsvm(x',y, 'ClassNames',[-1,1],...
    'KernelFunction','rbf', 'KernelScale', 1/sqrt(kappa_best), ...
    'BoxConstraint', 0.5/m/lam);

KSVM_worst = fitcsvm(x',y, 'ClassNames',[-1,1],...
    'KernelFunction','rbf', 'KernelScale', 1/sqrt(kappa_worst), ...
    'BoxConstraint', 0.5/m/lam);

%% Zeichnen der erlernten Hypothesen

x1 = -1.75 : 4 / 500 : 2.25; % Gitterpunkte in x1-Richtung
x2 = -1.75 : 4 / 500 : 2.25; % Gitterpunkte in x1-Richtung
[XX1,XX2] = meshgrid(x1,x2); % 2D-Gitter erzeugen
X1 = XX1(:); X2 = XX2(:); % Gitterpunkt-Matrizen als Spaltenvektoren

[~,grid_labels_best] = predict(KSVM_best, [X1,X2]);
[~,grid_labels_worst] = predict(KSVM_worst, [X1,X2]);

% Zeichnen:
figure(1)
hold on;

% Erlernte Trennlinien
contour(XX1,XX2,reshape(grid_labels_best(:,2),size(XX1)),[0 0],'-g','LineWidth',2);
contour(XX1,XX2,reshape(grid_labels_worst(:,2),size(XX1)),[0 0],'-m','LineWidth',2);

% Wahre Trennlinien
plot(x1, (x1.^2 - 3*x1 - 2*0)/3 ,'-k')
plot(x1, (x1.^2 - 3*x1 - 2*pi)/3 ,'-k')
plot(x1, (x1.^2 - 3*x1 + 2*pi)/3 ,'-k')

% Grafikparameter
ylim([-1.75,2.25])
legend({'-1','1','best h_S','worst h_S','truth'},'Location','Best');
title(['\kappa = ',sprintf('%f', kappa)])
xlabel('x_1')
ylabel('x_2')
set(gca,'fontsize',14)


%% d) Vergleich CV-Fehler zu geschätzten Verallgemeinerungsfehler

% Neue Daten erzeugen
x_neu = 4*rand(2,m) - 1.75; % zufaellige x-Werte in [-1.75,2.253]^2
% Wahre multivariate Funktion fuer Klassifizierung:
f_true = @(x) sin( 0.5*( x(1,:).^2 - 3*(x(2,:) + x(1,:)) ) ); 
% Entsprechende Markierungen:
y_neu = 2*(f_true(x_neu) >0)-1;

error_gen = mean(y_neu' ~=  predict(KSVM_best, x_neu'))
CV_min

error_gen = mean(y_neu' ~=  predict(KSVM_worst, x_neu'))
CV_max

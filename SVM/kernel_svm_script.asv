% MATHEMATIK DES MASCHINELLEN LERNENS
%===========================================
% Kapitel 3: Lineare Klassifikationsmethoden
% Abschnitt 3.3: Harte SVM-Regel
%
% MATLAB-Skript zum Nachvollziehen des 
% Beispiels zur harten SVM-Regel

% Alles loeschen und schliessen
clear all; close all; clc;

%% (0) Vorbereitung
%------------------ 

% Laden der Trainingsdaten
load('data_KSVM');

% AUSKOMMENTIERT:
% % Oder: Erzeugen zufaelliger Daten
% m = 3e2; % Anzahl der Daten
% x = 4*rand(2,m) - 1.75; % zufaellige x-Werte in [-1.75,2.253]^2
% 
% % Wahre multivariate Funktin fuer Klassifizierung:
% f_true = @(x) sin( 0.5*( x(1,:).^2 - 3*(x(2,:) + x(1,:)) ) ); 
% 
% % Entsprechende Markierungen:
% y = 2*(p(x) >0)-1;

% Zeichnen der Trainingsdaten:
indp = find(y>0); % Daten mit +1 
indm = find(y<0); % Daten mit -1 
figure(1);
plot(x(1,indp),x(2,indp),'ob','Linewidth',2); hold on;
plot(x(1,indm),x(2,indm),'+r','Linewidth',2); hold on;
grid on;
axis tight;
xlabel('x_1')
ylabel('x_2')
set(gca,'fontsize',14)

%% (1) Weiche Kern-SVM-Regel (Gauss-Kern)
%----------------------------------------

% Lambda fuer weiche Regel:
m = length(y);
lam = 0.5/m;

% Skalierungsparameter
kappa = 4;

% Aufruf der Routine aus der Machine-Learing-Toolbox
KSVM = fitcsvm(x',y, 'ClassNames',[-1,1],...
    'KernelFunction','rbf', 'KernelScale', 1/sqrt(kappa), ...
    'BoxConstraint', 0.5/m/lam);

% HINWEIS: 'BoxConstraint' entspricht 1/(2*m*lambda), 'KernelScale' ist
% gerade 1/sqrt(kappa) -- siehe Hilfe fuer fitcsvm



%% (2) Zeichnen der erlernten Hypothese
%----------------------------------------

% Dazu folgende Diskretisierung
x1 = -1.75 : 4 / 500 : 2.25; % Gitterpunkte in x1-Richtung
x2 = -1.75 : 4 / 500 : 2.25; % Gitterpunkte in x1-Richtung
[XX1,XX2] = meshgrid(x1,x2); % 2D-Gitter erzeugen
X1 = XX1(:); X2 = XX2(:); % Gitterpunkt-Matrizen als Spaltenvektoren


%% (2.1) Einzeichnen der Trennlinien

% Vorhergesagte Labels an Gitterpunkte
[~,grid_labels] = predict(KSVM, [X1,X2]);
% grid_labels = predict(KSVM, [X1,X2]);
% grid_labels

% Zeichnen:
figure(1); hold on;

% Support vectors
plot(x(1,KSVM.IsSupportVector), x(2,KSVM.IsSupportVector),'ko','MarkerSize',10);

% Erlernte Trennlinien
contour(XX1,XX2,reshape(grid_labels(:,2),size(XX1)),[0 0],'-g','LineWidth',2);

% Wahre Trennlinien
plot(x1, (x1.^2 - 3*x1 - 2*0)/3 ,'-k')
plot(x1, (x1.^2 - 3*x1 - 2*pi)/3 ,'-k')
plot(x1, (x1.^2 - 3*x1 + 2*pi)/3 ,'-k')

% Grafikparameter
ylim([-1.75,2.25])
legend({'-1','1','Support Vectors','h_S','truth'},'Location','Best');
title(['\kappa = ',sprintf('%f', kappa)])
xlabel('x_1')
ylabel('x_2')
set(gca,'fontsize',14)

%% (2.2) Zeichnen der RKHS-Funktion

% Support-Vektoren finden: 
supp_ind = find(KSVM.IsSupportVector); % Indizes der SV
m_supp = length(supp_ind); % Anzahl der SV
SV_supp = KSVM.SupportVectors'; % Koordinaten der SV


% Erlernte Koeffizienten:
alpha_S = KSVM.Alpha .* y(supp_ind)'; % alpha-Vektor
b_S = KSVM.Bias; % Bias
kappa_S =1/KSVM.KernelParameters.Scale^2; % Skalenparameter

% Distanzfunktion zu den Supportvektoren:
SV_fun = @(x) sum((SV_supp- repmat(x,1,m_supp)).^2,1);

% Erlernte RKHS-Funktion:
Kfun_S = @(x) exp(- kappa_S * SV_fun(x)) * alpha_S + b_S;

X1
X2
[X1(1); X2(1)]
length(X1)
% Werte der RKHS-Funktion auf dem Gitter
Z = X1;
for i = 1:length(X1),
    Z(i) = Kfun_S([X1(i); X2(i)]);
end

% Surface-Plot:
figure(2);
surf(XX1,XX2, reshape(Z,length(x1),length(x2)),'Edgecolor','none');
axis tight;
xlabel('x_1')
ylabel('x_2')
zlabel('f_{S}')
set(gca,'fontsize',14)
view([40,25])

% 2D-Plot:
figure(3)
C = max(Z); % Maximum der RKHS-Funktion zum Verschieben
surf(XX1,XX2,reshape(double(Z-C),length(x1),length(x2)),'Edgecolor','none')
hold on;
% Niveaulinien
contour(XX1,XX2, reshape(double(Z),length(x1),length(x2)), [0 0], 'LineWidth',2)
% Datenpunkte:
plot3(x(1,indp), x(2,indp), ones(length(indp),1),'ob','Linewidth',2);
plot3(x(1,indm), x(2,indm), ones(length(indm),1),'+r','Linewidth',2);
% Support-Vektoren
plot(SV_supp(1,:),SV_supp(2,:),'ko','MarkerSize',10);
% Grafik-Parameter
view(2)
xlabel('x_1')
ylabel('x_2')
set(gca,'fontsize',14)

% 
% %% (3) Kappa-Studie (Gauss-Kern)
% %----------------------------------------
% 
% % Lambda fuer weiche Regel:
% m = length(y);
% lam = 0.5/m;
% 
% % Skalierungsparameter
% kappa = 4;
% kappa = 1/(0.075)^2;
% %kappa = 1/(3^2);
% 
% % Aufruf der Routine aus der Machine-Learing-Toolbox
% KSVM_kappa = fitcsvm(x',y, 'ClassNames',[-1,1],...
%     'KernelFunction','rbf', 'KernelScale', 1/sqrt(kappa), ...
%     'BoxConstraint', 0.5/m/lam);
% 
% % Vorhergesagte Labels an Gitterpunkten
% [~,grid_labels] = predict(KSVM_kappa, [X1,X2]);
% 
% % Zeichnen:
% figure(); hold on;
% plot(x(1,indp),x(2,indp),'ob','Linewidth',2); hold on;
% plot(x(1,indm),x(2,indm),'+r','Linewidth',2); hold on;
% plot(x(1,KSVM_kappa.IsSupportVector), x(2,KSVM_kappa.IsSupportVector),'ko','MarkerSize',10);
% % Erlernte Trennlinien
% contour(XX1,XX2,reshape(grid_labels(:,2),size(XX1)),[0 0],'-k','LineWidth',2);
% % Grafikparameter
% ylim([-1.75,2.25])
% %legend({'-1','1','Support Vectors','h_S','truth'},'Location','Best');
% title(['\kappa = ',sprintf('%.3f', kappa)])
% xlabel('x_1')
% ylabel('x_2')
% set(gca,'fontsize',14)
% 
% %% (4) Lambda-Studie (Gauss-Kern)
% %----------------------------------------
% 
% % Lambda fuer weiche Regel:
% m = length(y);
% lam = 0.5/m;
% lam = 1/(2*m)^2;
% lam = 0.035;
% 
% % Skalierungsparameter
% kappa = 4;
% 
% % Aufruf der Routine aus der Machine-Learing-Toolbox
% KSVM_lam = fitcsvm(x',y, 'ClassNames',[-1,1],...
%     'KernelFunction','rbf', 'KernelScale', 1/sqrt(kappa), ...
%     'BoxConstraint', 0.5/m/lam);
% 
% % Vorhergesagte Labels an Gitterpunkten
% [~,grid_labels] = predict(KSVM_lam, [X1,X2]);
% 
% % Zeichnen:
% figure(); hold on;
% plot(x(1,indp),x(2,indp),'ob','Linewidth',2); hold on;
% plot(x(1,indm),x(2,indm),'+r','Linewidth',2); hold on;
% plot(x(1,KSVM_lam.IsSupportVector), x(2,KSVM_lam.IsSupportVector),'ko','MarkerSize',10);
% % Erlernte Trennlinien
% contour(XX1,XX2,reshape(grid_labels(:,2),size(XX1)),[0 0],'-k','LineWidth',2);
% % Grafikparameter
% ylim([-1.75,2.25])
% %legend({'-1','1','Support Vectors','h_S','truth'},'Location','Best');
% title(['\lambda = ',sprintf('%f', lam)])
% xlabel('x_1')
% ylabel('x_2')
% set(gca,'fontsize',14)
% 
% %% (5) Polynomieller Kern
% %------------------------
% 
% lam = 0.5/m; % Default-Wert
% kappa = 1; % Default-Wert
% q = 3; % Default-Wert
% 
% % Aufruf der Routine aus der Machine-Learing-Toolbox
% KSVM_poly = fitcsvm(x',y, 'ClassNames',[-1,1],...
%     'KernelFunction','polynomial', 'KernelScale', 1/sqrt(kappa), ...
%     'PolynomialOrder', q)
% 
% %% (5.1) Zeichnen der Trennlinien
% [~,grid_labels] = predict(KSVM_poly, [X1,X2]);
% 
% figure();
% 
% % Daten
% plot(x(1,indp),x(2,indp),'ob','Linewidth',2); hold on;
% plot(x(1,indm),x(2,indm),'+r','Linewidth',2);
% 
% % Support vectors
% plot(x(1,KSVM_poly.IsSupportVector), x(2,KSVM_poly.IsSupportVector),'ko','MarkerSize',10);
% 
% % Erlernte Trennlinien
% contour(XX1,XX2,reshape(grid_labels(:,2),size(XX1)),[0 0],'-g','LineWidth',2);
% 
% % Wahre Trennlinien
% plot(x1, (x1.^2 - 3*x1 - 2*0)/3 ,'-k')
% plot(x1, (x1.^2 - 3*x1 - 2*pi)/3 ,'-k')
% plot(x1, (x1.^2 - 3*x1 + 2*pi)/3 ,'-k')
% 
% % Grafikparameter
% ylim([-1.75,2.25])
% legend({'-1','1','Support Vectors','h_S','truth'},'Location','Best');
% title('Polynomial Kernel (q=3)')
% xlabel('x_1')
% ylabel('x_2')
% set(gca,'fontsize',14)
% 
% %% (5.2) Zeichnen der RKHS-Funktion
% 
% % Support-Vektoren finden: 
% supp_ind = find(KSVM_poly.IsSupportVector); % Indizes der SV
% m_supp = length(supp_ind); % Anzahl der SV
% X_supp = KSVM_poly.SupportVectors'; % Koordinaten der SV
% 
% 
% % Erlernte Koeffizienten:
% alpha_S = KSVM_poly.Alpha .* y(supp_ind)'; % alpha-Vektor
% b_S = KSVM_poly.Bias; % Bias
% kappa_S = 1/KSVM_poly.KernelParameters.Scale^2; % Skalenparameter
% 
% % Innenproduktfunktion zu den Supportvektoren:
% SV_fun = @(x) sum(X_supp .* repmat(x,1,m_supp),1);
% 
% % Erlernte RKHS-Funktion:
% Kfun_S = @(x) (1 + kappa_S * SV_fun(x) ).^q * alpha_S + b_S;
% 
% % Werte der RKHS-Funktion auf dem Gitter
% Z = X1;
% for i = 1:length(X1),
%     Z(i) = Kfun_S([X1(i); X2(i)]);
% end
% 
% % Surface-Plot
% figure();
% surf(XX1,XX2, reshape(Z,length(x1),length(x2)),'Edgecolor','none');
% axis tight;
% xlabel('x_1')
% ylabel('x_2')
% zlabel('f_{S}')
% set(gca,'fontsize',14)
% view([40,25])
% 
% % 2D-Plot:
% figure()
% C = max(Z); % Maximum der RKHS-Funktion zum Verschieben
% surf(XX1,XX2,reshape(double(Z-C),length(x1),length(x2)),'Edgecolor','none')
% hold on;
% % Niveaulinien
% contour(XX1,XX2, reshape(double(Z),length(x1),length(x2)), [0 0], 'LineWidth',2)
% % Datenpunkte:
% plot3(x(1,indp), x(2,indp), ones(length(indp),1),'ob','Linewidth',2);
% plot3(x(1,indm), x(2,indm), ones(length(indm),1),'+r','Linewidth',2);
% % Support-Vektoren
% plot(X_supp(1,:),X_supp(2,:),'ko','MarkerSize',10);
% % Grafik-Parameter
% view(2)
% xlabel('x_1')
% ylabel('x_2')
% set(gca,'fontsize',14)
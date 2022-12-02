% MATHEMATIK DES MASCHINELLEN LERNENS
%===========================================
% 5. Uebungsblatt - Aufgabe 1b
%
% MATLAB-Skript zum Ausfuehren des Kern-
% Perzeptron-Algorithmus

% Alles loeschen und schliessen
clear all; close all; clc;

% Laden der Trainingsdaten
load('data_KSVM');
m = length(y);
indp = find(y>0); % Daten mit +1 
indm = find(y<0); % Daten mit -1 

%% (1) Kern-Perzeptron-Algorithmus anwenden

% Wahl der Gauss-Kern-Funktion:
kappa = 4;
K_fun = @(x,y) exp(-kappa * norm(x-y)^2); 

% Erstellen der Gramschen Matrix bzgl. der Gauss-Kernfunktion
tic;
K = zeros(m,m);
for i = 1:m,
    for j = 1:m,
        K(i,j) = K_fun(x(:,i),x(:,j));
    end
end
toc;

% HINWEIS: Schnellere Variante:
tic;
e = ones(m,1); % Vektor aus lauter Einsen
% Matrix der paarweise quadrierten euklidschen Abstaende
R2 = (e * x(1,:) - x(1,:)' * e').^2 + (e * x(2,:) - x(2,:)' * e').^2;
K = exp(-kappa * R2);
toc;


% KPA anwenden:
[alpha_S, b_S, T, isSV] = my_kernel_perceptron(K,y);


%% (2) Zeichnen des Ergebnisses
%------------------------------

% Dazu folgende Diskretisierung
x1 = -1.75 : 4 / 500 : 2.25; % Gitterpunkte in x1-Richtung
x2 = -1.75 : 4 / 500 : 2.25; % Gitterpunkte in x1-Richtung
[XX1,XX2] = meshgrid(x1,x2); % 2D-Gitter erzeugen
X1 = XX1(:); X2 = XX2(:); % Gitterpunkt-Matrizen als Spaltenvektoren

% Support-Vektoren finden: 
m_supp = sum(isSV); % Anzahl der SV
SV_supp = x(:,isSV); % Koordinaten der SV

% Distanzfunktion zu den Supportvektoren:
SV_fun = @(x) sum((SV_supp - repmat(x,1,m_supp)).^2,1);

% Erlernte RKHS-Funktion:
Kfun_S = @(x) exp(- kappa * SV_fun(x)) * alpha_S(isSV) + b_S;

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
% view([40,25])

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
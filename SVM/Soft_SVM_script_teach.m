% MATHEMATIK DES MASCHINELLEN LERNENS
%===========================================
% Kapitel 3: Lineare Klassifikationsmethoden
% Abschnitt 3.4: Weiche SVM-Regel
%
% MATLAB-Skript zum Nachvollziehen des 
% Beispiels zur weichen SVM-Regel

% Alles loeschen und schliessen
clear all; close all; clc;

%% (0) Vorbereitung
%------------------ 

% Erzeugen der Trainingsdaten
m = 25; % Anzahl der Daten
x = 6*rand(2,m)-3; % zufaellige x-Werte in [-3,3]^2

w_true = [1; 2]; % wahre trennende Hyperebene

% Wahrscheinlichkeiten fuer Label +1 gemaess des Bernoulli-Modells mit
% h_{w_true,0}
p = 1./(1+exp(-(w_true' * x))) ; 

% Auswuerfeln der zufaelligen Markierungen gemaess der Wahrscheinlichkeiten
% p
y = 2*(rand(1,m) <= p)-1; 

% Bzw. Laden bereits generierter Daten:
load('data_svm_soft'); m = length(y);

%% (1) Weiche SVM-Regel
%----------------------------

lam = 1/m; % beide Terme gleichgewichtet

% Verlustfunktion definieren
hinge = @(w,x,y) max( [1-y.*(w'*x); zeros(1,m)], [], 1);

% Die Lösung bestimmen
fun = @(w) lam * norm(w).^2 + mean(hinge(w,x,y));
w_S = fminunc(fun, [0;0]);


%% (1.1) Zeichen der Zielfunktion

% Wir diskretisieren w1 und w2
w1 = ( -10*abs(w_S(1)) : 20*abs(w_S(1))/1000 : 10*abs(w_S(1)))';
w2 = ( -10*abs(w_S(2)) : 20*abs(w_S(2))/1000 : 10*abs(w_S(2)))';

% Zeichnen der Hoehenlinien der Zielfunktion
[WW1,WW2] = meshgrid(w1,w2); % Diskretisierungsgitter erzeugen
W1 = WW1(:); W2 = WW2(:);
FW = zeros(length(W1),1);

for i = 1:length(W1),
    ww = [W1(i); W2(i)];
    FW(i) = fun(ww);
end
    
% Grafik erzeugen (Contour-Plot):
figure(2); hold off;
contour(WW1,WW2,reshape(log(FW),length(w1),length(w2)),25,'LineWidth',2);
colorbar; % Farblegende
hold on;
plot(w_S(1),w_S(2),'or','LineWidth',2) % erlernten Wert einzeichnen

% Weitere Grafikparameter
xlabel('w_1')
ylabel('w_2')
axis tight
title(sprintf('log(%.2f |w|^2 + R_S(w))',lam))
set(gca,'FontSize',18)

%% (2) Zeichnen der Trainingsdaten
% --------------------------------

figure(1); hold off;

% Erst die "wahre" Hyperebene fuer x in [-3,3] einzeichnen
plot( [-3,3], -w_true(1)/w_true(2)*[-3,3], '--k','Linewidth',2) ;

% Einzeichnen der erlernten Hypothese in die Grafik der Daten
plot( [-3,3], -w_S(1)/w_S(2)*[-3,3], '-g','Linewidth',2)

% Zum Vergleich: logistische Regression einzeichnen
RS_log = @(w) mean( log(1 + exp(- y .* (w' * x))),2);
w_LR = fminunc(RS_log, [0;0]);
plot( [-3,3], -w_LR(1)/w_LR(2)*[-3,3], '-','Linewidth',2,'Color',[0.9290, 0.6940, 0.1250])

% Dann die klassifizierten Punkte eintragen
indp = find(y==1); % Punkte mit Markierung 1
plot(x(1,indp),x(2,indp),'b+','linewidth',2); hold on;

indm = find(y==-1); % Punkte mit Markierung -1
plot(x(1,indm),x(2,indm),'rd','linewidth',2); hold on;

% Weitere Werte der Grafik setzen
xlim([-3,3])
ylim([-3,3])
grid on
%axis equal 
axis tight
xlabel('x_1')
ylabel('x_2')
legend({'"Wahr"','Soft SVM','Log-Reg'},'Location','NorthWest')
set(gca,'FontSize',18)


% MATHEMATIK DES MASCHINELLEN LERNENS
%===========================================
% Kapitel 3: Lineare Klassifikationsmethoden
% Abschnitt 3.2: Logistische Regression
%
% MATLAB-Skript zum Nachvollziehen des 
% Beispiels fuer die logistische Regression

%% (0) Vorbereitung
%------------------ 

% Erzeugen der Trainingsdaten
m = 50; % Anzahl der Daten

x = 6*rand(2,m)-3; % zufaellige x-Werte in [-3,3]^2
x

w_true = [1; 2]; % wahre Parameter
w_true
% Wahrscheinlichkeiten fuer Label +1 gemaess des Bernoulli-Modells mit
% h_{w_true,0}
p = 1./(1+exp(-(w_true' * x))) ; 
p

% Auswuerfeln der zufaelligen Markierungen gemaess der Wahrscheinlichkeiten
% p
y = 2*(rand(1,m) <= p)-1; 
y

%% (1) Zeichnen der Trainingsdaten
% --------------------------------

figure(1); hold off

% Erst die wahre Hyperebene fuer x in [-3,3] einzeichnen
plot( [-3,3], -w_true(1)/w_true(2)*[-3,3], '--k','Linewidth',2); hold on 

% Dann die klassifizierten Punkte eintragen
indp = find(y==1); % Punkte mit Markierung 1
plot(x(1,indp),x(2,indp),'b+','linewidth',2); hold on;

indm = find(y==-1); % Punkte mit Markierung -1
plot(x(1,indm),x(2,indm),'rd','linewidth',2); hold on;

% Weitere Werte der Grafik setzen
xlim([-3,3])
ylim([-3,3])
grid on
axis equal 
axis tight
xlabel('x_1')
ylabel('x_2')
set(gca,'FontSize',18)

%% (2) Logistische Regression
%----------------------------

% Empirische Risiko-Funktion

% HIER: Die Funktionsdefinition ergaenzen:
x_y = x .* repmat(y, 2, 1);
RS = @(w) mean(log(1./(1+exp(-(w' * x_y)))), 2);
% log(1 + exp(- Y_train .* (X_train * w(1:d) + w(end))))

% HINWEISE: 
% 1) w ist ein Spaltenvektor mit zwei Zeilen ODER eine Matrix mit zwei
% Zeilen und mehreren Spalten fuer mehrere Gewichtsvektoren w
% 2) mean(. ,2) bildet den Mittelwert pro Spalte
% 3) Die Funktion sollte fuer mehrere w-Vektoren auswertbar sein, so dass
% das Ergebnis als Zeilenvektor die empirischen Risikowerte der einzelnen
% w-Vektoren beinhaltet.

% Zeichnen dieser Funktion
ws = -5:0.01:10; % Diskretisierung der w-Werte pro Achse
[WW1,WW2] = meshgrid(ws,ws); % Diskretisierungsgitter erzeugen
RS_Ws = RS([WW1(:)'; WW2(:)']); % RS an den Gitterpunkte auswerten


% Grafik erzeugen (Contour-Plot):
figure(2); hold off;
contour(WW1,WW2,reshape(RS_Ws,length(ws),length(ws)),25,'LineWidth',2);
colorbar; % Farblegende

% Weitere Grafik-Parameter setzen
xlabel('w_1')
ylabel('w_2')
title('log R_S(h_w)')
axis tight;
view(2);
set(gca,'FontSize',18)

% Numerische Berechnung der ERM-Parameter
[w,RS_min] = fminunc(RS, [0;0]);

% Einzeichnen der erlernten Parameter
figure(2); hold on;
plot(w(1),w(2),'or','LineWidth',2) % erlernt
plot(w_true(1),w_true(2),'+k','LineWidth',2) % wahr

% % Einzeichnen der erlernten Geraden in Figure 1:
% figure(1); hold on;
% plot( [-3,3], -w(1)/w(2)*[-3,3], '-g','Linewidth',2)
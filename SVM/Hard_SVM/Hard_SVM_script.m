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

% Erzeugen der Trainingsdaten
rng(17);
m = 25; % Anzahl der Daten
% x = 6*rand(2,m)-3; % zufaellige x-Werte in [-3,3]^2
x = [[-1.23200998  0.18352053 -1.85087528 -2.59259785  1.72191276  0.93800113 0.82512538  0.45361736 -2.7656225  -0.85311837  2.67409912 -2.63973192 2.18425262  2.26374316 -2.69283801  0.91451169  0.31050821  0.58507952 -0.09882825 -1.30207103 -1.21364569  0.36905343 -0.62371538  1.73220426 -0.48909369]; 
    [-2.13657648 -2.09455983 -2.6685519   1.30822316 -1.24609587 -1.80735677 1.98818353  0.40794672 -2.5059615   0.26999144 -2.04624753  1.06057431 -2.28916641 -0.3300236   2.3278946   1.78360638 -2.59212763  2.76472654 0.95523235  1.3126565   1.46146059  2.32641811 -2.19632599  1.66184748 2.02794907]];

w_true = [1; 2]; % wahre trennende Hyperebene
y = sign(w_true' * x) + (w_true' * x == 0); % wahre Markierungen

% Randabstand der wahren Hypothese
gamma_true = 1/norm(w_true) * min( abs(w_true' * x) )

%% (1) Harte SVM-Regel
%----------------------------

% Wir bestimmen die Matrix A fuer die linearen Nebenbedingungen
A = repmat(y,2,1) .* x;
% Die Lösung bestimmen
fun = @(w) norm(w).^2;
[w, fw] = fmincon( fun, [0;0], -A', -ones(m,1))

% Maximaler Randabstand
gamma = 1/sqrt(fw)


% Wir zeichnen den zulaessigen Bereich und die Zielfunktion

% Dazu bestimmen wir die Spalten von A mit A(2,.) negativ bzw. positiv ...
indAn = find(A(2,:) < 0);
indAp = find(A(2,:) > 0);

% ... und bauen uns daraus die untere und obere Schranke des zulaessigen 
% Bereiches
a_low = @(w1) max( (1 - w1 * A(1,indAp))./repmat(A(2,indAp),length(w1),1), [], 2);
a_up = @(w1) min( (1 - w1 * A(1,indAn))./repmat(A(2,indAn),length(w1),1),[], 2);

% Wir diskretisieren w1 im entsprechenden Bereich
w1 = ( 0 : 0.01 : 1/gamma)';

% Und bestimmen die Grenzen fuer w2
w2_low = a_low(w1);
% w2_low
w2_up = a_up(w1);

% Wir suchen den "Schnittpunkt" der unteren und oberen Grenze
ind = min(find(w2_low <= w2_up));

% Zeichnen der Hoehenlinien der Zielfunktion
w1s = ( 0 : max(w1)/500 : max(w1))'; % Diskretisierung der w-Werte pro Achse
w2s = ( 0 : max(w2_up)/500 : max(w2_up))'; % Diskretisierung der w-Werte pro Achse
[WW1,WW2] = meshgrid(w1s,w2s); % Diskretisierungsgitter erzeugen
F_Ws = WW1(:).^2 + WW2(:).^2; % RS an den Gitterpunkte auswerten

% Grafik erzeugen (Contour-Plot):
figure(2); hold off;
contour(WW1,WW2,reshape(F_Ws,length(w1s),length(w2s)),25,'LineWidth',2);
colorbar; % Farblegende

% Wir zeichnen die zulaessige Menge ein
figure(2); hold on;
fill([w1(ind:end);w1(end:-1:ind)],[[w2_low(ind:end);w2_up(end:-1:ind)]] ,'m')
% Wir zeichen die Loesung der harten SVM-Regel und den wahren Parameter ein
hold on;
plot(w(1),w(2),'or','LineWidth',2) % erlernt
plot(w1, w_true(2)/w_true(1)*w1 ,'--k') % wahre Ebene

% Weitere Grafikparameter
xlim([0,1/gamma])
ylim([0,max(w2_up)])
xlabel('w_1')
ylabel('w_2')
axis tight
names{1} = 'Hoehenlinien |w|^2';
names{2} = 'Zulaessiger Bereich';
names{3} = 'w_s';
names{4} = 'wahre Ebene';
legend(names,'Location','NorthWest')
set(gca,'FontSize',18)

%% (2) Zeichnen der Trainingsdaten
% --------------------------------

figure(1); hold off;

% Erst die wahre Hyperebene fuer x in [-3,3] einzeichnen
plot( [-3,3], -w_true(1)/w_true(2)*[-3,3], '--k','Linewidth',2) ;

% Einzeichnen der erlernten Hypothese in die Grafik der Daten
figure(1); hold on
plot( [-3,3], -w(1)/w(2)*[-3,3], '-g','Linewidth',2)

plot(x(1,11),x(2,11),'ok','LineWidth',2)

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
names2{1} = 'Wahre Ebene';
names2{2} = 'Erlernte Ebene';
legend(names2,'Location','NorthWest')
set(gca,'FontSize',18)
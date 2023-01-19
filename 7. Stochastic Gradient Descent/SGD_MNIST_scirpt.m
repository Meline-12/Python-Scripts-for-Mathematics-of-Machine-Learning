% MATHEMATIK DES MASCHINELLEN LERNENS
%===========================================
%
% 8. Übung: Stochastische Gradientenverfahren
%

% Alles loeschen und schliessen
clear all; close all; clc;

%% (0) Vorbereitung

% Laden der Daten:
load('data_MNIST_78.mat'); 
Y = (Y == 7) - (Y == 8);

m = length(Y); % Anzahl Daten
d = size(X,1); % Anzahl Merkmale (= Dimension des Merkmalsraums)


%% (1) Gradientenverfahren fuer Log-Verlust
%----------------------------------------

% Hilfsgrößen zur schnelleren Berechnung von y*(w*x+b):
X1 = [X;ones(1,m)];
X1Y = repmat(Y',size(X1,1),1) .* X1;
exp_XY = @(w) exp(-  w' * X1Y )';

% Empirisches Log-Risiko als Funktion von e = exp(- y*(w*x+b))
RS = @(e) mean( log(1 + e ), 1);

% Gradient des empirischen Log-Risikos wieder als Funktion von 
% e = exp(- y*(w*x+b))
Grad_RS = @(e) - (X1 * ( (Y .* e)./(1+e) ) )/m;

% Lipschitz-Konstante des Gradienten abschätzen gemäß Vorlesung
L = 1/4 * mean( sum(X.*X,1) );

% Maximal erlaubte Schrittweite laut Vorlesung
eta = 1/L;

% Gradientenverfahren
n_iter = 10; % Anzahl der Schritte
%n_iter=m ;
ws = zeros(d+1,n_iter+1); % Matrix der Iterierten
ws(:,1) = [zeros(d,1); 1]; % Startpunkt w_0


% Zeitmessung über 10 Schritte
tic;
for i =1:n_iter,
    % Berechnung von exp(- y*(w*x+b))
    e = exp_XY(ws(:,i));
    % Gradientenschritt
    ws(:,i+1) = ws(:,i) - eta * Grad_RS(e);
end
toc;

% Berechnung der empirischen Risiken für alle Iterierten
Fs = RS(exp_XY(ws));  

% Zeichnen der Funktion
figure();
semilogx(Fs,'-','LineWidth',2)
grid on;
xlabel('Schritt k')
ylabel('F(w_k) = R_S(w_k)')
hold on;
set(gca,'FontSize',16)

%figure();
%plot(Fs,'-','LineWidth',2)
%grid on;
%xlabel('Schritt k')
%ylabel('F(w_k) = R_S(w_k)')
%hold on;
%set(gca,'FontSize',16)

%% (2) Stochastisches Gradientenverfahren

% Anzahl der Schritte und Schrittweiten:
n_iter_SGD = m;
eta_k = @(k) 0.5 / (1 + k) ;

ws_SGD = zeros(d+1,n_iter_SGD+1); % Matrix der Iterierten
ws_SGD(:,1) = [zeros(d,1); 1]; % Startpunkt w_0


% Zeitmessung über m Schritte
tic;
for i =1:n_iter_SGD,
    ind = randperm(m,1); % Zufälligen Datenpunkt auswählen
    x = X1(:,ind); % Entsprechendes Merkmal x
    y = Y(ind); % Entsprechendes Label y
    e = exp(-  y * ws_SGD(:,i)' * x ); % Berechnung von exp(- y*(w*x+b))
    v = - (y * e /(1+e)) * x; % Richtung des Gradienten für Datenpunkt (x,y)
    
    % Gradientenschritt
    ws_SGD(:,i+1) = ws_SGD(:,i) - eta_k(i) * v;
end
toc;

% Berechnung der empirischen Risiken
Fs_SGD = RS(exp_XY(ws_SGD));

semilogx(Fs_SGD,'--','LineWidth',2)
%plot(Fs_SGD,'--','LineWidth',2)
%% Skript zur Auswertung der mittels Perzeptron erlernten Hypothese für MNIST
%
%

%% Daten laden
clear all; % alles löschen
load('data_MNIST_78'); % Bilder (X) und Label (Y) laden

% Transfomieren der Labels in +1 (7) und -1 (8):
Y = (Y == 7) - (Y == 8);

%% Starten des Perzeptron-Algorithmus
tic; 
[w, T, ws, RSs] = my_perceptron(X, Y', 1, 2000); 
toc;

% Defintion bzw. Ausgabe der gelernten Hypothese (in 7 und 8):
h_S = @(x) 7.5 - 0.5 * sign(w' * [x; 1]);

%% (a) Wieviele Falschklassifikationen gibt es im Trainingsdatensatz ?
m = length(Y); % Anzahl Daten
check = @(w,x,y) y .* (w' * [x; ones(1, m)]); % Check-Funktion
% Anteil falsch klassifizierter Bilder:
mean(check(w, X, Y') <= 0)

%% (b) Beispiele fuer falsch klassifizierte Bilder finden

ind = find(check(w, X, Y') <= 0); % finde Indizes falsch klassifizierter Bilder

% Zeichne erstes und letztes dieser Bilder
figure();
x = X(:,ind(1));
imshow(reshape(x,28,28)',[0,1]);
title(sprintf('Erkannt als: %i', h_S(x)))

figure();
x = X(:,ind(end));
imshow(reshape(x,28,28)',[0,1]);
title(sprintf('Erkannt als: %i', h_S(x)))

%% (c) Wähle richtig klassifiertes Bild aus

ind = find(check(w, X, Y') > 0); % finde Indizes richtig klassifizierter Bilder

% Zufällige Auswahl durch zufällige Komponente von ind
i = ind( randperm(length(ind), 1) )

% Bild zeichnen
x = X(:,i);
y = Y(i);
figure();
imshow(reshape(x, 28,28)',[0,1]);
title(sprintf('Erkannt als: %i', h_S(x)))

% Bestimme Störungsrichtung v (je nach Label von x)
if y > 0,
    v = w .* (w < 0); % negative Teil von w
else
    v = w .* (w > 0); % positiver Teil von w
    
end

v = v(1:end-1); % Bias weglassen

% Finde richtige Skalierung e:
e = - 1.1 * (w' * [x; 1]) / (w(1:end-1)' * v);

% Ueberpruefen, ob danach tatsächlich falsch klassifiert
y * (w' * [x + e*v; 1]) < 0
    

% Plotten der Bilder und Stoerungen
figure();
subplot(1,4,1); imshow(reshape(x, 28,28)',[0,1]); 
title(sprintf('Erkannt als: %i', h_S(x))); colorbar();

subplot(1,4,2); imshow(reshape(x+e*v, 28,28)',[0,1]); 
title(sprintf('Erkannt als: %i', h_S(x+e*v))); colorbar();

subplot(1,4,3); imshow(reshape(e*v, 28,28)',[0,1]); 
title({'Differenz'}); colorbar();

subplot(1,4,4); imshow(reshape(e*v, 28,28)',[0,max(e*v)]); 
title({'Differenz gezoomt'}); colorbar();
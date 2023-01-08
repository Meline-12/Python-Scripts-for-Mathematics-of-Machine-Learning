% MATHEMATIK DES MASCHINELLEN LERNENS
%===========================================
% Uebungsblatt 4: Aufgabe 4
%

% Alles loeschen und schliessen
clear all; close all; clc;

%% a) Vorbereiten der Daten
%-------------------------- 

% Laden des Datensatzes
T = readtable('heart.dat');

% Datensatz zu numerischer Matrix machen
T = T{:,:};
T

% Extrahieren der reellen Merkmale
X = T(:,[1,4,5,8,10,12])
X

% Extrahieren und Transformieren der Markierungen
Y = 2*T(:,14) - 3;
Y
% Anzahl der Datenpaare
m = length(Y);

% Anzahl der Merkmale
d = size(X,2);

%% b) Aufteilen der Daten
%-------------------------- 

% Zufaellige Auswahl der Indizes der Trainings- und Testdaten
p = 0.7; % Anteil der Trainingsdaten
ind_train = randperm(m, ceil(p*m));
ind_test = setdiff(1:m, ind_train);

% Trainingsdaten
X_train = X(ind_train,:); 
Y_train = Y(ind_train); 

% Testdaten
X_test = X(ind_test,:); 
Y_test = Y(ind_test);

%% c) Logistische Regression
%---------------------------

% HINWEIS: Wir nehmen im Vektor w den Bias an letzter Stelle mit auf.

% Empirische Risiko-Funktion
RS_log = @(w) mean( log(1 + exp(- Y_train .* (X_train * w(1:d) + w(end)))) , 1);

% Numerische Berechnung der ERM-Parameter...

% ... dazu erlauben wir genuegend Iteration...
options = optimoptions(@fminunc,'MaxFunctionEvaluations',1e5); % in python as parameter maxfun = 100 000

% ... und waehlen einen zufaelligen Startwert:
% 0.5*randn(d+1,1)
[w_LR, RS_min] = fminunc(RS_log, zeros(d+1,1), options);

% Bestimmen der falsch klassifizierten Trainingsdaten ueber Verletzung der NB:
Err_Train = mean( (Y_train .* (X_train * w_LR(1:d) + w_LR(end)) < 0)   , 1);
disp(sprintf('Es werden %.2f Prozent der Trainingsdaten falsch klassifiziert.\n',Err_Train));

% ANTWORT: Falls die Stichprobe linear trennbar waere, wuerde die
% logistische Regression die entsprechende trennende Hypothese finden.
% Aufgrund der vorhandenen Fehlklassifikationen ist dies also nicht der
% Fall.

% Bestimmen der falsch klassifizierten Testdaten ueber Verletzung der NB:
Err_Test = mean( (Y_test .* (X_test * w_LR(1:d) + w_LR(end)) < 0)   , 1);
disp(sprintf('Es werden %.2f Prozent der Testdaten falsch klassifiziert.\n',Err_Test));

% ANTWORT:
disp(sprintf('Wir schaetzen das erwartete Risiko von h_S also auf %.2f.\n',Err_Test));


%% d) Weiche SVM-Lernregel
%---------------------------

% Wahl von lamba:
lam = 1/m; % beide Terme gleich gewichtet

% Verlustfunktion definieren
hinge = @(w,x,y) max( [ 1 - y.*( x * w(1:d) + w(end) ), zeros(length(y),1)], [], 2);

% Die Lösung bestimmen
fun = @(w) lam * norm(w(1:d)).^2 + mean(hinge(w,X_train,Y_train));
%  options = optimoptions(@fminunc,'MaxFunctionEvaluations',1e5);
%  w_SVM = fminunc(fun, randn(d+1,1),options);
w_SVM = [ 0.85728827; -0.76466468; 0.06481023; 0.30725263; -0.04192087; -0.9862483; 1.5885821 ];
hinge(w_SVM, X_train,Y_train)
fun(w_SVM)
% Bestimmen der falsch klassifizierten Trainingsdaten ueber Verletzung der NB:
Err_Train = mean( (Y_train .* (X_train * w_SVM(1:d) + w_SVM(end)) < 0)   , 1);
disp(sprintf('Es werden %.2f Prozent der Trainingsdaten falsch klassifiziert.\n',Err_Train));

% Bestimmen der falsch klreassifizierten Testdaten ueber Verletzung der NB:
Err_Test = mean( (Y_test .* (X_test * w_SVM(1:d) + w_SVM(end)) < 0)   , 1);
disp(sprintf('Es werden %.2f Prozent der Testdaten falsch klassifiziert.\n',Err_Test));

% ANTWORT:
disp(sprintf('Wir schaetzen das erwartete Risiko von h_S also auf %.2f.\n',Err_Test));

% HINWEIS: Fuehren Sie diesen Abschnitt mehrmals aus und beobachten die
% Variationen der Ergebnisse aufgrund der zufaelligen Startwerte. Dies
% verdeutlicht, dass die numerische Optimierung nicht trivial ist und
% insbesondere nicht immer das globale Minimum findet.

% %% e) Lambda-Studie
% %------------------
% % Bereich der lambda-Werte:
% lams = 10.^(-3:0.1:1);
% n_lam = length(lams);
% 
% % Vektor der Fehler in den Trainings- und Testdaten
% Errs_Train = zeros(n_lam,1);
% Errs_Test = zeros(n_lam,1);
% 
% % Optionen fuer numerische Optimierung setzen:
% options = optimoptions(@fminunc,'Display','off','MaxFunctionEvaluations',1e5);
% 
% 
% % Schleife ueber diese Werte
% for i = 1:n_lam,
% lam = lams(i); % Lambda festlegen
% 
% % Zielfunktion neu definieren aufgrund des neuen Lambda-Wertes
% fun = @(w) lam * norm(w(1:d)).^2 + mean(hinge(w,X_train,Y_train));
% w_SVM = fminunc(fun, rand(d+1,1), options);
% 
% Errs_Train(i) = Errs_Train(i) + mean( (Y_train .* (X_train * w_SVM(1:d) + w_SVM(end)) < 0)   , 1);
% Errs_Test(i) = Errs_Test(i) + mean( (Y_test .* (X_test * w_SVM(1:d) + w_SVM(end)) < 0)   , 1);
% end
% 
% % Ergebnisse zeichnen
% figure();
% semilogx(lams, Errs_Test,'-or',lams, Errs_Train,'-.b','LineWidth',2)
% xlabel('\lambda')
% ylabel('Fehlklassifikationsraten')
% grid on;
% legend('Test','Training','Location','Best')
% set(gca,'FontSize',14)
% 
% % ANTWORT: Es gibt einen deutlichen Abfall der Fehler bei ca. lambda = 0.1 
% % (der genaue Werte kann von den zufaellig ausgewaehlten Daten abhaengen).
% % Fuer groessere Werte von lambda sind die Fehler deutlich hoeher, fuer
% % kleinere in etwa gleich. Insbesondere ist die Fluktuation innerhalb der
% % grossen Werte und kleinen Werte kleiner als der Abfall am "Schrankenwert"
% % von lambda, so dass man von zwei Regimen sprechen koennte.
% 
% % ANTWORT: (1) Die besten erzielten Werte fuer die Testdaten unterscheiden sich 
% % dabei nicht besonders von den Ergebnissen der logistischen Regression. 
% % (2) Aufgrund der einfacheren Optimierung und vielfaeltigeren
% % Interpretierbarkeit koennte man daher die log. Regression der weichen
% % SVM-Regel vorziehen. 
% % (3 )Der Vorteil der SVM-Regeln, die dimensionsunabhaengige Informations-
% % komplexitaet, kommt bei 6 Merkmalsdimensionen evtl. nocht nicht zum
% % Tragen.
% 
% %% Groessere Lambda-Studie um Zufallsfehler etwas auszuglaetten:
% 
% Errs_Train = Errs_Train/100;
% Errs_Test = Errs_Test/100;
% 
% % Schleife ueber 100 Durchlaeufe
% for j = 1:100,
% % Schleife ueber diese Werte
%     for i = 1:n_lam,
%     lam = lams(i); % Lambda festlegen
% 
%     % Zielfunktion neu definieren aufgrund des neuen Lambda-Wertes
%     fun = @(w) lam * norm(w(1:d)).^2 + mean(hinge(w,X_train,Y_train));
%     w_SVM = fminunc(fun, rand(d+1,1), options);
% 
%     Errs_Train(i) = Errs_Train(i) + mean( (Y_train .* (X_train * w_SVM(1:d) + w_SVM(end)) < 0)   , 1);
%     Errs_Test(i) = Errs_Test(i) + mean( (Y_test .* (X_test * w_SVM(1:d) + w_SVM(end)) < 0)   , 1);
%     end
% end
% Errs_Train = Errs_Train/100;
% Errs_Test = Errs_Test/100;
% 
% % Ergebnisse zeichnen
% figure();
% semilogx(lams, Errs_Test,'-or',lams, Errs_Train,'-.b','LineWidth',2)
% xlabel('\lambda')
% ylabel('Fehlklassifikationsraten')
% grid on;
% legend('Test','Training','Location','Best')
% set(gca,'FontSize',14)


% w_py = [-0.2472 0.2125 -0.0007 0.4610 0.0221 0.1490]';
% Err_Test = mean( (Y_test .* (X_test * w_py(1:d) + w_py(end)) < 0)   , 1);
% Err_Test
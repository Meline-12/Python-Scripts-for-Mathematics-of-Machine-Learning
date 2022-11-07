function [w,T,ws, RSs] = my_perceptron(x,y,b,n_iter)
%
% Diese Funktion soll den Perzeptron-Algorithmus aus Abschnitt 3.1
% ausfuehren. Dabei soll mittels des dritten (optionalen) Arguments
% unterschieden werden, ob eine homogene lineare Hypothese gelernt werden
% soll.
%
% INPUT:
% x     ...     (d,m)-Matrix bestehend aus den m Trainingsmerkmalen im R^d
% y     ...     (1,m)-Vektor bestehend aus den m zugehoerigen Labels -1,+1
% b     ...     optionale Variable, die fuer den Wert 0 eine homogene
%               lineare Hypothese aus den Daten lernt, sonst eine allg.
%               lineare Hypothese (default)
% n_iter ...    Maximale Anzahl an Interationen fuer den Algorithmus (optional)
%
% OUTPUT:
% w     ...     Spaltenvektor, der die erlernten Gewichte und ggf. Bias
%               enthaelt in der Form (w_1, w_2, ... w_d, b)^T
% T     ...     Integer der Anzahl der ausgefuehrten Schritte im Algorithmus
% ws    ...     Matrix mit T+1 Spalten, die t-te Spalte enthaelt die t-te 
%               Iterierte des Verfahrens
% RSs   ...     Zeilenvektor, der das empirische Risiko zu jedem Vektor ws
%               enthält
%

% Auslesen der Dimension d und der Datenanzahl m aus x bzw. y
d = size(x,1);
m = length(y);

% Falls n_iter nicht angegeben, wird es auf unendlich gesetzt
if nargin < 4,
    n_iter = Inf;
end

% Fallunterscheidung, ob homogene Hypothese gelernt werden soll
if nargin < 3 | b == 1, 
    % Der Fall der allgemeinen affin-linearen Hypothese
    
    % Erweiterten Gewichtsvektor initialisieren
    w = zeros(d+1,1);
    
    % Erster Eintrag in ws:
    ws = w;
    
    % Funktion zum Ueberprufen der Nebenbedingungen
    check = @(w,x,y) y .* (w' * [x; ones(1,m)]);
    
    % Berechnung des erzielten empirischen Risikos
    RS = @(w) mean(check(w,x,y) <= 0);
    
    % Empirisches Risiko des aktuellen w:
    RSs(1) = RS(w);
    
    % Iterationen über while-Schleife
    t = 0;
    while min(check(w,x,y)) <= 0 & t < n_iter
        
        % Alle nichterfuellte Nebenbedingungen finden 
        ...
            
        % Eine nichterfuellte Nebenbedingung auswaehlen
        ...
        
        % Update gemaess Iterationsvorschrift
        ...
        
        % Aktuelles w in ws speichern
        ...
            
        % Empirisches Risiko berechnen und in RSs speichern
        ...
        
        % Schrittzaehler erhoehen
        t = t+1;
        
    end
else
    % Der Fall der homogenen linearen Hypothese mit b = 0

    % Gewichtsvektor etc. initialisieren
    w = zeros(d,1);
    
    % Erster Eintrag in ws:
    ws = w;
    
    % Funktion zum Ueberprufen der Nebenbedingungen (jetzt ohne b!)
    check = @(w,x,y) y .* (w' * x);

    % Berechnung des erzielten empirischen Risikos
    RS = @(w) mean(check(w,x,y) <= 0);
    
    % Empirisches Risiko des aktuellen w:
    RSs(1) = RS(w);

    % Iterationen über while-Schleife
    t = 0;
    while min(check(w,x,y)) <= 0 & t < n_iter
        
        % Alle nichterfuellte Nebenbedingungen finden 
        ...
            
        % Eine nichterfuellte Nebenbedingung auswaehlen
        ...
        
        % Update gemaess Iterationsvorschrift
        ...
        
        % Aktuelles w in ws speichern
        ...
            
        % Empirisches Risiko berechnen und in RSs speichern
        ...
        
        % Schrittzaehler erhoehen
        t = t+1;
    end
end

% Schrittanzahl ausgeben
T = t;

end
function [alpha,b,T, isSV] = my_kernel_perceptron(K,y)
%
% Diese Funktion soll den Kern-Perzeptron-Algorithmus vom 5. Übungsblatt
% ausfuehren. 
%
% INPUT:
% K     ...     (m,m)-Matrix bestehend aus der Gramschen Matrix der
%               gewaehlten Kernfunktion ausgewertet an den Merkmalsdaten
% y     ...     (1,m)-Vektor bestehend aus den m zugehoerigen Labels -1,+1
%
% OUTPUT:
% alpha ...     (m,1)-Spaltenvektor, der die erlernten Koeffizienten des 
%                Gewichtsvektors w bzgl. der Kernfunktion K(x_i, . ) beinhaltet
% b     ...     Erlerntes Bias-Wert
% T     ...     Integer der Anzahl der ausgefuehrten Schritte im Algorithmus
% isSV  ...     (m,1)-Spaltenvektor bestehend aus logischen Werten, die 
%               angeben, ob der i-te Datenpunkt ein Stuetzvektor ist, also
%               ob alpha_i ungleich Null ist
%

% Anzahl m der Daten aus y oder K auslesen:
m = length(y);

% Initialisierungen:
alpha = zeros(m,1);
b = 0;
t = 0;

alphas = [alpha;b];

% Ueberpruefen, ob alle Nebenbedingungen erfuellt sind
check = y' .* (K*alpha+b);

% while-Schleife bis alle Nebenbedingungen erfuellt sind
while min(check) <= 0
    % Finden bzw. auswaehlen einer nichterfuellten Nebenbedingung:
    is = find(check <=0 );
    i = is( randsample(length(is),1) );
    
    % Update gemaess der Iterationsvorschrift
    alpha(i) = alpha(i) + y(i);
    b = b + y(i);
    alphas(:,t+1) = [alpha;b];
    
    % Ueberpruefen der Nebenbedingung und Schrittzaehler erhoehen:
    check = y' .* (K*alpha+b);
    t = t+1;
end

% Ausgabewerte festlegen:
T = t;
isSV = abs(alpha) > 0;
end


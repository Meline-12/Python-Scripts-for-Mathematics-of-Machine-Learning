load('data_KSVM');

y = y(1:10);
x = x(:, 1:10);
m = length(y);
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


% Anzahl m der Daten aus y oder K auslesen:
m = length(y);

% Initialisierungen:
alpha = zeros(m,1);
b = 0;
t = 0;

alphas = [alpha;b];
alphas

% Ueberpruefen, ob alle Nebenbedingungen erfuellt sind
check = y' .* (K*alpha+b);

% while-Schleife bis alle Nebenbedingungen erfuellt sind
while min(check) <= 0 && t<2
    % Finden bzw. auswaehlen einer nichterfuellten Nebenbedingung:
    is = find(check <=0 );
    i = is( randsample(length(is),1) );
    
    % Update gemaess der Iterationsvorschrift
    alpha(i) = alpha(i) + y(i);
    alpha
    b = b + y(i);
    b
    alphas(:,t+1) = [alpha;b];
    alphas
    % Ueberpruefen der Nebenbedingung und Schrittzaehler erhoehen:
    check = y' .* (K*alpha+b);
    check
    t = t+1;
end

% Ausgabewerte festlegen:
T = t;
isSV = abs(alpha) > 0;


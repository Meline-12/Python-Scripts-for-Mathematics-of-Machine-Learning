% MATHEMATIK DES MASCHINELLEN LERNENS
%===========================================
% Kapitel 3: Lineare Klassifikationsmethoden
% Abschnitt 3.1: Das Perzeptron
%
% MATLAB-Skript zum Nachvollziehen des 
% Beispiels zum Perzeptron-Algorithmus

%% (0) Vorbereitung
%------------------ 

% Erzeugen der Trainingsdaten
m = 25; % Anzahl der Daten
x = 6*rand(2,m)-3; % zufaellige x-Werte in [-3,3]^2  random.uniform Matrix 2 rows, m colums, high low insteed of [0, 1] should be [-3, 3] 6 and -3 
x
w_true = [1; 2]; % wahre trennende Hyperebene
w_true
y = sign(w_true' * x) + (w_true' * x == 0); % wahre Markierungen # w_true*x == 0 we want to have as 1 # w_true' * x == 0 np.dot(w_true.T, x)
y
%% (1) Zeichnen der Trainingsdaten
% --------------------------------

figure(1); 

% Erst die wahre Hyperebene fuer x in [-3,3] einzeichnen
plot( [-3,3], -w_true(1)/w_true(2)*[-3,3], '--k','Linewidth',2) 

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

%% (2) Perzeptron-Algorithmus
%----------------------------

% Den Algorithmus auf die Daten anwenden mit b = 0 fest
[w,T,ws] = my_perceptron(x,y,0,100);

% Einzeichnen der erlernten Hypothese in die Grafik der Daten
figure(1); hold on
plot( [-3,3], -w(1)/w(2)*[-3,3], '-g','Linewidth',2)

%%%%%%%%%%%%%%%%%%%%%%

% Wir bestimmen die Matrix A um den Kegel der Loesungen zu zeichnen und B
% numerisch zu berechnen:
A = repmat(y,2,1) .* x;

% Wir bestimmen die Spalten von A mit A(2,.) negativ bzw. positiv
indAn = find(A(2,:) < 0);
indAp = find(A(2,:) > 0);

% Wir finden die begrenzenden Anstiege des Loesungskegels
a_low = max(-A(1,indAp)./A(2,indAp));
a_up = min(-A(1,indAn)./A(2,indAn));

% Wir legen die Koordinaten des zu zeichnenden Polygons fuer den
% abgeschnitten Kegel fest
area_x = [0, 6, 6, 0];
area_y = [0, 6*a_low, 6*a_up, 0];

% Wir zeichnen des Loesungskegel und die Iterierten des
% Perzeptron-Algorithmuses
figure(2);
fill(area_x,area_y,'g') % Zeichne ausgefuellten Kegel

% Zeichne Iterierten des Algorithmuses ein
hold on;
plot(ws(1,:),ws(2,:),'o-k','Linewidth',2);

% Setze weitere Werte der Grafik und fuege Legende hinzu
xlim([0,6])
ylim([0,10])
xlabel('w_1')
ylabel('w_2')
axis tight
names{1} = 'Loesungsmenge';
names{2} = 'Iterationen des Perzeptron';
legend(names,'Location','NorthWest')
set(gca,'FontSize',18)

% Wir berechnen noch R und B
R = max( sqrt(x(1,:).^2 + x(2,:).^2) )

% B wird mittels numerischer Optimierung bestimmt
fun = @(w) norm(w);
[~,B] = fmincon( fun, [0;0], -A', -ones(m,1))

% HINWEIS: fun wird minimiert, der Startvektor ist w =(0,0) und die
% Nebenbedingung ist -A^T * w \leq -1


%% Mit Legende zeichnen
figure(1);
hold off; 
plot( [-3,3], -w_true(1)/w_true(2)*[-3,3], '--k','Linewidth',2); hold on
plot( [-3,3], -w(1)/w(2)*[-3,3], '-g','Linewidth',2)
plot(x(1,indp),x(2,indp),'b+','linewidth',2); hold on;
plot(x(1,indm),x(2,indm),'rd','linewidth',2); hold on;
xlim([-3,3])
ylim([-3,3])
xlabel('x_1')
ylabel('x_2')
grid on
axis tight
set(gca,'FontSize',18)
names{1} = 'Wahre Trennebene';
names{2} = 'Erlernte Trennebene';
legend(names,'Location','NorthEast')

R = max( sqrt(x(1,:).^2 + x(2,:).^2) )
[~,B] = fmincon(@(w) norm(w), [0;0], -A', -ones(m,1))



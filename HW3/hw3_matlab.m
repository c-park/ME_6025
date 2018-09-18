% Intro to Optimization
% Homework 3
% Cade Parkison

P = [1 7; 2 6; 5 8; 7 7; 9 5; 3 7];
B = [(P.*P)*[1 1]', P(:,1), P(:,2), ones(6, 1)];
[U,S,V] = svd(B);
d = diag(S);
res = V(:, find(d == min(d)));
xc = -res(2)/(2*res(1));
yc = -res(3)/(2*res(1));
r = sqrt((res(2)^2 + res(3)^2)/(4*res(1)^2) - res(4)/res(1));
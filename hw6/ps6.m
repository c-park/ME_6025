% Problem Set 6

% Problem 3 with CVX

rand('seed', 314);
x = rand(40, 1); y = rand(40, 1);
class = [ 2*x < y+0.5 ] + 1 ;
A1 = [ x(find(class==1)) , y(find(class==1)) ] ;
A2 = [ x(find(class==2)) , y(find(class==2)) ] ;
figure(1 ); hold on ;
plot(A1(:, 1),A1(: ,2), '*', 'MarkerSize', 6 );
plot(A2(:, 1),A2(:,2), 'd', 'MarkerSize', 6 );


cvx_begin
    variables a(2) b(1)
    A1'a - b >= 1;
    A2'a - b <= -1;
cvx_end

L = 1;
lc = 0.1;
//+
Point(1) = {0, 0, 0, lc};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, L, 0, lc};
Point(4) = {0, L, 0, lc};
//+
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Periodic Curve {4} = {2} Translate {-L, 0, 0};
//+
Line Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Curve(10) = {1};
Physical Curve(20) = {2};
Physical Curve(30) = {3};
Physical Curve(40) = {4};
Physical Surface("mesh") = {1};

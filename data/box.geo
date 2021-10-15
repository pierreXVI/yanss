L = 10;
lc = L / 50;
//+
RECOMBINE = 0;
STRUCTURED = 1;
TRANSFINITE = 0;
//+
Point(1) = {-L/2, -L/2, 0, lc};
Point(2) = { L/2, -L/2, 0, lc};
Point(3) = { L/2,  L/2, 0, lc};
Point(4) = {-L/2,  L/2, 0, lc};
//+
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
//+
Line Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
If (STRUCTURED || RECOMBINE)
  Recombine Surface {1};
EndIf
If (STRUCTURED || TRANSFINITE)
  Transfinite Surface {1};
EndIf
//+
Physical Curve(10) = {1};
Physical Curve(30) = {3};
// Periodic Curve {3} = {1} Translate( 0, L, 0);
Physical Curve(20) = {2};
Physical Curve(40) = {4};
// Periodic Curve {4} = {2} Translate(-L, 0, 0);
Physical Surface("mesh") = {1};

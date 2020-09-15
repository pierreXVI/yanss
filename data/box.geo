L = 0.1;
lc = 0.01;
//+
RECOMBINE = 1;
STRUCTURED = 1;
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
//+
Line Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
If (STRUCTURED)
  Transfinite Curve {1, 3} = (1 + L / lc) Using Progression 1;
  Transfinite Curve {2, 4} = (1 + L / lc) Using Progression 1;
  Transfinite Surface {1};
EndIf
If (STRUCTURED || RECOMBINE)
  Recombine Surface {1};
EndIf
//+
Physical Curve(10) = {1};
Physical Curve(20) = {2};
Physical Curve(30) = {3};
Physical Curve(40) = {4};
Physical Surface("mesh") = {1};

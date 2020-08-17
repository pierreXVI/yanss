AIRFOIL = 0012;  // NACA airfoil name
CHORD = 1;       // Cord length
RESOLUTION = 20; // Half the number of points used to mesh the airfoil

MSH_SIZE_INNER = 1 * (CHORD / RESOLUTION);  // Size of the mesh on the surface of the airfoil
MSH_SIZE_OUTTER = 0.1;  // Size of the mesh on the outter border

HEIGHT = 0.5 * CHORD; // Height of the outter bountary
LENGTH = 1.2 * CHORD; // Length of the outter bountary

// ====================================

M = Floor(AIRFOIL / 1000) / 100;
P = Floor(AIRFOIL % 1000 / 100) / 10;
T = Floor(AIRFOIL % 100);
Macro y_naca
  If (M > 0 && P > 0)
    If (x < P)
      yc = M * (2*P*x - x^2) / P^2;
    Else
      yc = M * (1 - 2*P + 2*P*x - x^2) / (1 - P)^2;
    EndIf
  Else
    yc = 0;
  EndIf
  a1 =  0.2969 * Sqrt(x);
  a2 = -0.1260 * x;
  a3 = -0.3516 * x^2;
  a4 =  0.2843 * x^3;
  a5 = -0.1015 * x^4;
  /* a5 = -0.1036 * x^4; // for a closed trailing edge */
  yt = (a1 + a2 + a3 + a4 + a5) * T / 20;
Return

pts[0] = newp; Point(pts[0]) = {0, 0, 0, MSH_SIZE_INNER};
For k In {1:2*RESOLUTION - 1}
  If (k < RESOLUTION)
    x = k / RESOLUTION;
    Call y_naca;
    y = yc + yt;
  Else
    x = 2 - k / RESOLUTION;
    Call y_naca;
    y = yc - yt;
  EndIf
  pts[k] = newp; Point(pts[k]) = {x * CHORD, y * CHORD, 0, MSH_SIZE_INNER};
  lines[k - 1] = newl; Line(lines[k - 1]) = {pts[k - 1], pts[k]};
EndFor
lines[2*RESOLUTION - 1] = newl; Line(lines[2*RESOLUTION - 1]) = {pts[2 * RESOLUTION - 1], pts[0]};


p1 = newp; Point(p1) = {0,       HEIGHT, 0, MSH_SIZE_OUTTER};
p2 = newp; Point(p2) = {0,      -HEIGHT, 0, MSH_SIZE_OUTTER};
p3 = newp; Point(p3) = {LENGTH, -HEIGHT, 0, MSH_SIZE_OUTTER};
p4 = newp; Point(p4) = {LENGTH,  HEIGHT, 0, MSH_SIZE_OUTTER};
c1 = newl; Circle(c1) = {p1, pts[0], p2};
l2 = newl; Line(l2) = {p2, p3};
l3 = newl; Line(l3) = {p3, p4};
l4 = newl; Line(l4) = {p4, p1};


naca_ll = newll; Line Loop(naca_ll) = lines[];    Physical Curve(10) = lines[];      // WALL
out_ll = newll; Line Loop(out_ll) = {l2, l3, l4}; Physical Curve(20) = {l2, l3, l4}; // OUTFLOW
in_ll = newll; Line Loop(in_ll) = {c1};           Physical Curve(30) = {c1};         // INFLOW

mesh_s = news; Plane Surface(mesh_s) = {naca_ll, in_ll, out_ll};
Physical Surface("mesh") = {mesh_s};

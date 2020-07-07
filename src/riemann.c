#include "physics.h"


// Number of iterations of the fixed point pressure solver for the Riemann problem
#define N_ITER_RIEMANN 10
// Epsilon of the pressure solver for the Riemann problem
#define EPS_RIEMANN 1E-14


void RiemannSolver_Euler_Exact(PetscInt dim, PetscInt Nf,
                               const PetscReal x[], const PetscReal n[], const PetscReal uL[], const PetscReal uR[],
                               PetscInt numConstants, const PetscScalar constants[], PetscReal flux[], void *ctx){
  Physics phys = (Physics) ctx;

  PetscFunctionBeginUser;

  PetscReal area = 0, nn[dim]; // n = area * nn, nn normal unitary vector to the surface
  for (PetscInt i = 0; i < dim; i++) area += PetscSqr(n[i]);
  area = PetscSqrtReal(area);
  for (PetscInt i = 0; i < dim; i++) nn[i] = n[i] / area;

  PetscReal wL[Nf], wR[Nf];
  ConservativeToPrimitive(phys, uL, wL);
  ConservativeToPrimitive(phys, uR, wR);

  PetscReal rl = wL[0], rr = wR[0];                    // densities
  PetscReal ul = 0, ur = 0;                            // normal speeds
  for (PetscInt i = 0; i < dim; i++){
    ul += wL[1 + i] * nn[i];
    ur += wR[1 + i] * nn[i];
  }
  PetscReal utl[dim], utr[dim];                        // tangent speeds
  for (PetscInt i = 0; i < dim; i++){
    utl[i] = wL[1 + i] - ul * nn[i];
    utr[i] = wR[1 + i] - ur * nn[i];
  }
  PetscReal pl = wL[dim + 1], pr = wR[dim + 1];        // pressures

  PetscReal cl = PetscSqrtReal(phys->gamma * pl / rl); // speeds of sound
  PetscReal cr = PetscSqrtReal(phys->gamma * pr / rr);

  PetscReal alpha = (phys->gamma + 1) / (2 * phys->gamma);
  PetscReal beta  = (phys->gamma - 1) / (2 * phys->gamma);
  PetscReal delta = (phys->gamma + 1) / (phys->gamma - 1);

  PetscReal pstar = PetscPowReal((beta * (ul - ur) + cl + cr) / (cl * PetscPowReal(pl, -beta) + cr * PetscPowReal(pr, -beta)), 1 / beta);
  PetscReal ml, mr;
  PetscBool flg_sl, flg_sr;
  for (PetscInt i = 0; i < N_ITER_RIEMANN; i++) {
    PetscReal pratiol = pstar / pl;
    if (pratiol > 1) {
      ml = rl * cl * PetscSqrtReal(1 + alpha * (pratiol - 1));
      flg_sl = PETSC_TRUE;
    } else if (pratiol < 1 - EPS_RIEMANN) {
      ml = rl * cl * beta * (1 - pratiol) / (1 - PetscPowReal(pratiol, beta));
      flg_sl = PETSC_FALSE;
    } else {
      ml = rl * cl * (3 - pratiol) / 2; // Linearisation of the rarefaction formula
      flg_sl = PETSC_FALSE;
    }

    PetscReal pratior = pstar / pr;
    if (pratior > 1) {
      mr = rr * cr * PetscSqrtReal(1 + alpha * (pratior - 1));
      flg_sr = PETSC_TRUE;
    } else if (pratior < 1 - EPS_RIEMANN) {
      mr = rr * cr * beta * (1 - pratior) / (1 - PetscPowReal(pratior, beta));
      flg_sr = PETSC_FALSE;
    } else {
      mr = rr * cr * (3 - pratior) / 2; // Linearisation of the rarefaction formula
      flg_sr = PETSC_FALSE;
    }

    pstar = (mr * pl + ml * pr + ml * mr * (ul - ur)) / (ml + mr);
    if (pstar < 0) {SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_NOT_CONVERGED, "ERROR : p* < 0");}
  }

  PetscReal ustar = (ml * ul + mr * ur + pl - pr) / (ml + mr);
  PetscReal rout, uout, *utout, pout; // density, normal and tangent speeds, and pressure at the interface
  if (ustar > 0) {
    utout = utl;
    PetscReal pratiol = pstar / pl;
    if (flg_sl) { // shock -> uL*
      uout = ustar;
      pout = pstar;
      rout = rl * (1 + delta * pratiol) / (delta + pratiol);
    } else {
      PetscReal rstarl = rl * PetscPowReal(pratiol, 1 / phys->gamma);
      PetscReal cstarl = PetscSqrtReal(phys->gamma * pstar / rstarl);
      if (ustar < cstarl) { // rarefaction -> uL*
        uout = ustar;
        pout = pstar;
        rout = rstarl;
      } else { // sonic rarefaction
        uout = (ul * cstarl - ustar * cl) / (ul - ustar + cstarl - cl);
        pout = pl * PetscPowReal(uout / cl, 1 / beta);
        rout = rl * PetscPowReal(pout / pl, 1 / phys->gamma);
      }
    }
  } else {
    utout = utr;
    PetscReal pratior = pstar / pr;
    if (flg_sr) { // shock -> uR*
      uout = ustar;
      pout = pstar;
      rout = rr * (1 + delta * pratior) / (delta + pratior);
    } else {
      PetscReal rstarr = rr * PetscPowReal(pratior, 1 / phys->gamma);
      PetscReal cstarr = PetscSqrtReal(phys->gamma * pstar / rstarr);
      if (ustar < cstarr) { // rarefaction -> uR*
        uout = ustar;
        pout = pstar;
        rout = rstarr;
      } else { // sonic rarefaction
        uout = (ur * cstarr - ustar * cr) / (ur - ustar + cstarr - cr);
        pout = pr * PetscPowReal(uout / cr, 1 / beta);
        rout = rr * PetscPowReal(pout / pr, 1 / phys->gamma);
      }
    }
  }

  PetscReal un = uout * area, unorm2 = 0;
  for (PetscInt i = 0; i < dim; i++) unorm2 += PetscSqr(uout * nn[i] + utout[i]);

  flux[0] = rout * un;
  for (PetscInt i = 0; i < dim; i++) flux[1 + i] = rout * (uout * nn[i] + utout[i]) * un + pout * n[i];
  flux[dim + 1] = (pout * phys->gamma / (phys->gamma - 1) + 0.5 * rout * unorm2) * un;

  PetscFunctionReturnVoid();
}

/*
void RiemannSolver_Euler_Roe(PetscInt dim, PetscInt Nf,
                                const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[],
                                PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx){
  Physics phys = (Physics) ctx;

  PetscFunctionBeginUser;

  PetscReal area = 0, nn[dim]; // n = area * nn, nn normal vector to the surface
  for (PetscInt i = 0; i < dim; i++){area += PetscSqr(n[i]);}
  area = PetscSqrtReal(area);
  for (PetscInt i = 0; i < dim; i++){nn[i] = n[i] / area;}


  PetscReal rl = uL[0], rr = uR[0]; // left and right densities
  PetscReal ul = 0, ur = 0; // left and right normal speeds
  for (PetscInt i = 0; i < dim; i++){
    ul += uL[1 + i] * nn[i];
    ur += uR[1 + i] * nn[i];
  }
  PetscReal utl[dim], utr[dim]; // left and right tangent speeds
  for (PetscInt i = 0; i < dim; i++){
    utl[i] = uL[1 + i] - ul * nn[i];
    utr[i] = uR[1 + i] - ur * nn[i];
  }
  PetscReal pl = uL[dim + 1], pr = uR[dim + 1]; // left and right pressures

  PetscReal cl = PetscSqrtReal(phys->gamma * pl / rl); // left speed of sound
  PetscReal cr = PetscSqrtReal(phys->gamma * pr / rr); // right speed of sound

  PetscReal alpha = PetscSqrtReal(rr / rl);
  PetscReal uROE[Nf];
  for (PetscInt i = 0; i < Nf; i++) {uRoe[i] = (uL[i] + alpha * uR[i]) / (1 + alpha);}


   // right hand side
  rr = max(face%primr(1,iface), cutoff)
  ur = face%primr(2,iface)
  vr = face%primr(3,iface)
  wr = face%primr(4,iface)
  pr = face%primr(5,iface)
  hp = gamma*pr*(1 / (phys->gamma - 1))/rr +HALF*(ur*ur+vr*vr+wr*wr)

  // left hand side
  rl = max(face%priml(1,iface), cutoff)
  ul = face%priml(2,iface)
  vl = face%priml(3,iface)
  wl = face%priml(4,iface)
  pl = face%priml(5,iface)
  hm = gamma*pl*(1 / (phys->gamma - 1))/rl +HALF*(ul*ul+vl*vl+wl*wl)

  // r  = sqrt(max(rr/rl, cutoff))
  // rr = sqrt(max(rr*rl, cutoff))
  alpha = PetscSqrtReal(rr / rl);
  beta = PetscSqrtReal(rr * rl);

  uu = (ur*r+ul)/(r+ONE)
  vv = (vr*r+vl)/(r+ONE)
  ww = (wr*r+wl)/(r+ONE)
  ee = HALF*(uu**2+vv**2+ww**2)
  hh = (hp*r+hm)/(r+ONE)
  cc = SQRT(max((phys->gamma - 1)*(hh-ee), cutoff))
  vn = uu*nn[0] + vv*nn[1]+ww*nn[2]

   // harten correction
  lambda1 = ABS(vn)
  lambda4 = ABS(vn+cc)
  lambda5 = ABS(vn-cc)

  // L. Tourrette formulation
  small = (2-global%HartenType)*global%HartenCoeff*(ABS(uu)+ABS(vv)+ABS(ww)+cc)  & ! Formulation 1 (harten_type = 1)
       + (global%HartenType-1)*global%HartenCoeff*(ABS(vn)+cc)                    ! Formulation 2 (harten_type = 2)

  q1 = HALF + SIGN(HALF,lambda1-small)
  k1 = HALF - SIGN(HALF,lambda1-small)
  q4 = HALF + SIGN(HALF,lambda4-small)
  k4 = HALF - SIGN(HALF,lambda4-small)
  q5 = HALF + SIGN(HALF,lambda5-small)
  k5 = HALF - SIGN(HALF,lambda5-small)

  oneonsmall = ONE/small
  mask = HALF + SIGN(HALF, small)
  aa1 = mask * (q1 * lambda1 + 0.5 * k1 * (lambda1 * lambda1 + small * small) / small) + (ONE - mask) * lambda1
  aa4 = mask * (q4*lambda4 &
      +k4*HALF*(lambda4*lambda4+small*small)*oneonsmall) + &
      (ONE-mask)*lambda4
  aa5 = mask * (q5*lambda5 &
      +k5*HALF*(lambda5*lambda5+small*small)*oneonsmall) + &
      (ONE-mask)*lambda5

  du  = ur-ul
  dv  = vr-vl
  dw  = wr-wl
  dvn = du*nn[0]+dv*nn[1]+dw*nn[2]
  dd1 = (rr-rl)-(pr-pl)/cc**2
  dd4 = HALF*(pr-pl+rr*cc*dvn)/cc**2
  dd5 = HALF*(pr-pl-rr*cc*dvn)/cc**2
  df12 = aa1*(dd1*uu+rr*(du-nn[0]*dvn))
  df13 = aa1*(dd1*vv+rr*(dv-nn[1]*dvn))
  df14 = aa1*(dd1*ww+rr*(dw-nn[2]*dvn))
  df15 = aa1*(dd1*ee+rr*(uu*du+vv*dv+ww*dw-vn*dvn))

  df452 = aa4*dd4*(uu+nn[0]*cc) + aa5*dd5*(uu-nn[0]*cc)
  df453 = aa4*dd4*(vv+nn[1]*cc) + aa5*dd5*(vv-nn[1]*cc)
  df454 = aa4*dd4*(ww+nn[2]*cc) + aa5*dd5*(ww-nn[2]*cc)
  df455 = aa4*dd4*(hh+vn*cc) + aa5*dd5*(hh-vn*cc)

  // Centred flux
  flux[0] = 0.5 * (uL[0] * unl + uR[0] * unr);
  for (PetscInt i = 0; i < dim; i++) {flux[1 + i] = 0.5 * (uL[1 + i] * unl + uR[1 + i] * unr + (pl + pr) * n[i]);}
  flux[dim + 1] = 0.5 * ((uL[dim + 1] + pl) * unl + (uR[dim + 1] + pr) * unr);

  // Correction
  flux[0] -= 0.5 * area * (aa1*dd1 + aa4*dd4 + aa5*dd5);
  flux[1 + 0] -= 0.5 * area * (df12 + df452);
  flux[1 + 1] -= 0.5 * area * (df13 + df453);
  flux[1 + 2] -= 0.5 * area * (df14 + df454);
  flux[5] -= 0.5 * area * (df15 + df455);


  PetscFunctionReturnVoid();
}
*/

void RiemannSolver_Euler_LaxFriedrichs(PetscInt dim, PetscInt Nf,
                                       const PetscReal x[], const PetscReal n[], const PetscReal uL[], const PetscReal uR[],
                                       PetscInt numConstants, const PetscScalar constants[], PetscReal flux[], void *ctx){
  Physics phys = (Physics) ctx;

  PetscFunctionBeginUser;
  SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_SUP, "LaxFriedrichs Riemann solver not implemented yet");

  PetscReal wL[Nf], wR[Nf];
  ConservativeToPrimitive(phys, uL, wL);
  ConservativeToPrimitive(phys, uR, wR);

  PetscReal coeff = 0;
  // PetscReal coeff = c / (2 * phys->cfl);

  PetscReal dotl = 0, dotr = 0;
  for (PetscInt i = 0; i < dim; i++) {
    dotl += wL[1 + i] * n[i];
    dotr += wR[1 + i] * n[i];
  }

  flux[0] = (uL[0] * dotl + uR[0] * dotr) / 2 - coeff * (uR[0] - uL[0]);
  for (PetscInt i = 0; i < dim; i++) flux[1 + i] = (uL[1 + i] * dotl + uR[1 + i] * dotr + (wL[dim + 1] + wR[dim + 1]) * n[i]) / 2 - coeff * (uR[1 + i] - uL[1 + i]);
  flux[dim + 1] = ((uL[dim + 1] + wL[dim + 1]) * dotl + (uR[dim + 1] + wR[dim + 1]) * dotr) / 2 - coeff * (uR[dim + 1] - uL[dim + 1]);

  PetscFunctionReturnVoid();
}

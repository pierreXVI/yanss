#include "physics.h"

void RiemannSolver_Euler_Exact(PetscInt dim, PetscInt Nf,
                                const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[],
                                PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx){
  Physics phys = (Physics) ctx;

  PetscFunctionBeginUser;

  PetscBool flag = (x[0] < 0.5);

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
  ul /= rl;
  ur /= rr;
  PetscReal utl[dim], utr[dim]; // left and right tangent speeds
  for (PetscInt i = 0; i < dim; i++){
    utl[i] = uL[1 + i] / rl - ul * nn[i];
    utr[i] = uR[1 + i] / rr - ur * nn[i];
  }
  PetscReal norm2l = 0, norm2r = 0;
  for (PetscInt i = 0; i < dim; i++){
    norm2l += PetscSqr(uL[1 + i]);
    norm2r += PetscSqr(uR[1 + i]);
  }
  PetscReal pl = (phys->gamma - 1) * (uL[Nf - 1] - 0.5 * norm2l / rl); // left pressure
  PetscReal pr = (phys->gamma - 1) * (uR[Nf - 1] - 0.5 * norm2r / rr); // right pressure

  if (flag) {PetscPrintf(PETSC_COMM_WORLD, "wL = %.3f, % .3f, %.1E ", rl, ul, pl);}
  if (flag) {PetscPrintf(PETSC_COMM_WORLD, "wR = %.3f, % .3f, %.1E ", rr, ur, pr);}

  PetscReal cl = PetscSqrtReal(phys->gamma * pl / rl); // left speed of sound
  PetscReal cr = PetscSqrtReal(phys->gamma * pr / rr); // right speed of sound

  PetscReal alpha = (phys->gamma + 1) / (2 * phys->gamma);
  PetscReal beta  = (phys->gamma - 1) / (2 * phys->gamma);
  PetscReal delta = (phys->gamma + 1) / (phys->gamma - 1);

  PetscReal pstar = PetscPowReal((beta * (ul - ur) + cl + cr) / (cl * PetscPowReal(pl, -beta) + cr * PetscPowReal(pr, -beta)), 1 / beta);
  PetscReal ml, mr;
  PetscBool flg_sl, flg_sr;
  for (PetscInt i = 0; i < N_ITER_RIEMANN; i++) {
    PetscReal pratiol = pstar / pl;
    PetscReal pratior = pstar / pr;
    flg_sl = pratiol >= 1;
    flg_sr = pratior >= 1;
    ml = (flg_sl) ? rl * cl * PetscSqrtReal(1 + alpha * (pratiol - 1)) : rl * cl * beta * (1 - pratiol) / (1 - PetscPowReal(pratiol, beta));
    mr = (flg_sr) ? rr * cr * PetscSqrtReal(1 + alpha * (pratior - 1)) : rr * cr * beta * (1 - pratior) / (1 - PetscPowReal(pratior, beta));
    pstar = (mr * pl + ml * pr + ml * mr * (ul - ur)) / (ml + mr);
    if (flag) {PetscPrintf(PETSC_COMM_WORLD, "pstar %f, %f\n", ml, mr);}
    if (pstar < 0) {SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "ERROR : p* < 0");}
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
  for (PetscInt i = 0; i < dim; i++) {unorm2 += PetscSqr(uout * nn[i] + utout[i]);}

  flux[0] = rout * un;
  for (PetscInt i = 0; i < dim; i++) {flux[1 + i] = rout * (uout * nn[i] + utout[i]) * un + pout * n[i];}
  flux[Nf - 1] = (pout * phys->gamma / (phys->gamma - 1) + 0.5 * rout * unorm2) * un;

  if (flag) {PetscPrintf(PETSC_COMM_WORLD, "Flux : % .1E, % .1E, % .1E, % .1E\n", flux[0], flux[1], flux[2], flux[3]);}

  PetscFunctionReturnVoid();
}

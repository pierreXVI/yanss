#include "physics.h"

/*
  Pointwise Riemann solver functions, with the following calling sequence:

  ```
  func(PetscInt dim, PetscInt Nf, const PetscReal x[], const PetscReal n[], const PetscScalar uL[],
       const PetscScalar uR[], PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx)
    dim          - Spatial dimension
    Nf           - Number of fields
    x            - Coordinates at a point on the interface
    n            - Area-scaled normal vector to the interface
    uL           - State vector to the left of the interface
    uR           - State vector to the right of the interface
    flux         - Output array of flux through the interface
    numConstants - Number of constant parameters
    constants    - Constant parameters
    ctx          - Context, to be casted to (Physics)
  ```
*/


/*
  Constant advection at speed 1 in the first direction, for debugging purposes
*/
static void RiemannSolver_AdvectionX(PetscInt dim, PetscInt Nc,
                              const PetscReal x[], const PetscReal n[], const PetscReal uL[], const PetscReal uR[],
                              PetscInt numConstants, const PetscScalar constants[], PetscReal flux[], void *ctx){
  Physics phys = (Physics) ctx;

  PetscFunctionBeginUser;

  const PetscReal un = phys->riemann_ctx.advection_speed * n[0];
  const PetscReal *u0 = (un < 0) ? uR : uL;
  for (PetscInt i = 0; i < Nc; i++) flux[i] = u0[i] * un;
  PetscFunctionReturnVoid();
}


/*
  From "I do like CFD", Katate Masatsuka
*/
static void RiemannSolver_Exact_PressureSolver_FP(PetscInt dim, Physics phys, PetscReal *wL, PetscReal *wR,
                                     PetscReal unL, PetscReal unR, PetscReal *utL, PetscReal *utR, PetscReal aL, PetscReal aR,
                                     PetscReal *pstar, PetscReal *ustar){
  PetscFunctionBeginUser;

  PetscReal alpha = (phys->gamma + 1) / (2 * phys->gamma); // = 1 - beta
  PetscReal beta  = (phys->gamma - 1) / (2 * phys->gamma); // = 1 - alpha

  // Initial guess: two rarefaction waves, not the best but insure next p* > 0
  *pstar = PetscPowReal((phys->gamma * beta * (unL - unR) + aL + aR) / (aL * PetscPowReal(wL[dim + 1], -beta) + aR * PetscPowReal(wR[dim + 1], -beta)), 1 / beta);

  PetscReal mL, mR;
  for (PetscInt i = 0; i < phys->riemann_ctx.pressure_solver_niter; i++) {
   PetscReal delta_p, pold = *pstar;

   { // Left wave
     PetscReal p_ratioL = *pstar / wL[dim + 1];
     if (p_ratioL > 1) {
       mL = wL[0] * aL * PetscSqrtReal(1 + alpha * (p_ratioL - 1));
     } else if (p_ratioL < 1 - phys->riemann_ctx.pressure_solver_eps) {
       mL = wL[0] * aL * beta * (1 - p_ratioL) / (1 - PetscPowReal(p_ratioL, beta));
     } else {
       mL = wL[0] * aL * (3 * phys->gamma - 1 + (phys->gamma + 1) * p_ratioL) / (4 * phys->gamma); // Linearisation of the rarefaction formula
     }
   }

   { // Right wave
     PetscReal p_ratioR = *pstar / wR[dim + 1];
     if (p_ratioR > 1) {
       mR = wR[0] * aR * PetscSqrtReal(1 + alpha * (p_ratioR - 1));
     } else if (p_ratioR < 1 - phys->riemann_ctx.pressure_solver_eps) {
       mR = wR[0] * aR * beta * (1 - p_ratioR) / (1 - PetscPowReal(p_ratioR, beta));
     } else {
       mR = wR[0] * aR * (3 - p_ratioR) / 2; // Linearisation of the rarefaction formula
     }
   }

   *pstar = (mR * wL[dim + 1] + mL * wR[dim + 1] + mL * mR * (unL - unR)) / (mL + mR);
   delta_p = *pstar - pold;
   if (*pstar < 0) SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_NOT_CONVERGED, "ERROR : p* < 0");
   if (2 * PetscAbs(delta_p) / (*pstar + pold) < phys->riemann_ctx.pressure_solver_rtol) break;
  }

  *ustar = (mL * unL + mR * unR + wL[dim + 1] - wR[dim + 1]) / (mL + mR);

  PetscFunctionReturnVoid();
}

/*
  From "Riemann Solvers and Numerical Methods for Fluid Dynamics", Euleterio F. Toro
*/
static void RiemannSolver_Exact_PressureSolver_Newton(PetscInt dim, Physics phys, PetscReal *wL, PetscReal *wR,
                                               PetscReal unL, PetscReal unR, PetscReal *utL, PetscReal *utR, PetscReal aL, PetscReal aR,
                                               PetscReal *pstar, PetscReal *ustar){
  PetscFunctionBeginUser;

  PetscReal alpha = (phys->gamma + 1) / (2 * phys->gamma); // = 1 - beta
  PetscReal beta  = (phys->gamma - 1) / (2 * phys->gamma); // = 1 - alpha
  PetscReal delta = (phys->gamma - 1) / (phys->gamma + 1); // = beta / alpha

  PetscReal fL, fR, delta_u = unR - unL;

  // Initial guess: two rarefaction waves, not the best but insure next p* > 0
  *pstar = PetscPowReal((aL + aR - phys->gamma * beta * delta_u) / (aL * PetscPowReal(wL[dim + 1], -beta) + aR * PetscPowReal(wR[dim + 1], -beta)), 1 / beta);

  for (PetscInt i = 0; i < phys->riemann_ctx.pressure_solver_niter; i++) {
    PetscReal fpL, fpR, delta_p, pold = *pstar;

    { // Left wave
      PetscReal p_ratioL = *pstar / wL[dim + 1];
      if (p_ratioL > 1) {
        PetscReal BL = delta * wL[dim + 1];
        PetscReal CL = PetscSqrtReal(1 / ((wL[0] * phys->gamma * alpha) * (wL[dim + 1] + BL)));
        fL = (pold - wL[dim + 1]) * CL;
        fpL = CL * (1 - (pold - wL[dim + 1]) / (2 * (wL[dim + 1] + BL)));
      } else {
        fL = (PetscPowReal(p_ratioL, beta) - 1) * aL / (phys->gamma * beta);
        fpL = PetscPowReal(p_ratioL, -alpha) / (wL[0] * aL);
      }
    }

    { // Right wave
      PetscReal p_ratioR = *pstar / wR[dim + 1];
      if (p_ratioR > 1) {
        PetscReal BR = delta * wR[dim + 1];
        PetscReal CR = PetscSqrtReal(1 / ((wR[0] * phys->gamma * alpha) * (wR[dim + 1] + BR)));
        fR = (pold - wR[dim + 1]) * CR;
        fpR = CR * (1 - (pold - wR[dim + 1]) / (2 * (wR[dim + 1] + BR)));
      } else {
        fR = (PetscPowReal(p_ratioR, beta) - 1) * aR / (phys->gamma * beta);
        fpR = PetscPowReal(p_ratioR, -alpha) / (wR[0] * aR);
      }
    }

    delta_p = (fL + fR + delta_u) / (fpL + fpR);
    *pstar = pold - delta_p;
    if (*pstar < 0) SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_NOT_CONVERGED, "ERROR : p* < 0");
    if (2 * PetscAbs(delta_p) / (*pstar + pold) < phys->riemann_ctx.pressure_solver_rtol) break;
  }

  *ustar = 0.5 * (unL + unR + fR - fL);

  PetscFunctionReturnVoid();
}

/*
  Exact Riemann solver
*/
static void RiemannSolver_Exact(PetscInt dim, PetscInt Nc,
                                const PetscReal x[], const PetscReal n[], const PetscReal uL[], const PetscReal uR[],
                                PetscInt numConstants, const PetscScalar constants[], PetscReal flux[], void *ctx){
  Physics phys = (Physics) ctx;

  PetscFunctionBeginUser;

  PetscReal alpha = (phys->gamma + 1) / (2 * phys->gamma); // = 1 - beta
  PetscReal beta  = (phys->gamma - 1) / (2 * phys->gamma); // = 1 - alpha
  PetscReal delta = (phys->gamma - 1) / (phys->gamma + 1); // = beta / alpha

  PetscReal area, nn[dim];
  { // n = area * nn, nn normal unitary vector to the surface
    area = 0;
    for (PetscInt i = 0; i < dim; i++) area += PetscSqr(n[i]);
    area = PetscSqrtReal(area);
    for (PetscInt i = 0; i < dim; i++) nn[i] = n[i] / area;
  }

  PetscReal wL[Nc], wR[Nc];
  { // Primitive variables (rho, u_1, ..., u_dim, p)
    ConservativeToPrimitive(phys, uL, wL);
    ConservativeToPrimitive(phys, uR, wR);
  }

  PetscReal unL, unR, utL[dim], utR[dim];
  { // Normal and tangent speeds
    unL = 0;
    unR = 0;
    for (PetscInt i = 0; i < dim; i++){
      unL += wL[1 + i] * nn[i];
      unR += wR[1 + i] * nn[i];
    }
    for (PetscInt i = 0; i < dim; i++){
      utL[i] = wL[1 + i] - unL * nn[i];
      utR[i] = wR[1 + i] - unR * nn[i];
    }
  }

  PetscReal aL, aR;
  { // Speeds of sound
    aL = PetscSqrtReal(phys->gamma * wL[dim + 1] / wL[0]);
    aR = PetscSqrtReal(phys->gamma * wR[dim + 1] / wR[0]);
  }

  PetscReal pstar, ustar;
  { // Solving pressure problem
    phys->riemann_ctx.pressure_solver(dim, phys, wL, wR, unL, unR, utL, utR, aL, aR, &pstar, &ustar);
  }

  PetscReal rho, un, *ut, p;
  { // Sampling the solution : density, normal and tangent speeds, and pressure at the interface
    if (ustar > 0) { // left side of contact
      ut = utL;
      PetscReal p_ratioL = pstar / wL[dim + 1];
      if (p_ratioL > 1) { // Shock wave
        if (unL - aL * PetscSqrtReal(alpha * p_ratioL + beta) < 0) { // Shock speed < 0, star state
          un = ustar;
          p = pstar;
          rho = wL[0] * (delta + p_ratioL) / (1 + delta * p_ratioL);
        } else { // Shock speed >= 0, left state
          un = unL;
          p = wL[dim + 1];
          rho = wL[0];
        }
      } else { // Rarefaction wave
        PetscReal rstarL = wL[0] * PetscPowReal(p_ratioL, 1 / phys->gamma);
        PetscReal astarL = PetscSqrtReal(phys->gamma * pstar / rstarL);
        if (unL - aL > 0) { // Left of the rarefaction
          un = unL;
          p = wL[dim + 1];
          rho = wL[0];
        } else if (ustar - astarL < 0) { // Right of the rarefaction
          un = ustar;
          p = pstar;
          rho = rstarL;
        } else { // In the rarefaction fan
          un = aL / (alpha * phys->gamma) + delta * unL;
          p = wL[dim + 1] * PetscPowReal(un / aL, 1 / beta);
          rho = wL[0] * PetscPowReal(un / aL, 1 / phys->gamma);
        }
      }
    } else { // Right side of the contact
      ut = utR;
      PetscReal p_ratioR = pstar / wR[dim + 1];
      if (p_ratioR > 1) { // Shock wave
        if (unR + aR * PetscSqrtReal(alpha * p_ratioR + beta) > 0) { // Shock speed > 0, star state
          un = ustar;
          p = pstar;
          rho = wR[0] * (delta + p_ratioR) / (1 + delta * p_ratioR);
        } else { // Shock speed <= 0, right state
          un = unR;
          p = wR[dim + 1];
          rho = wR[0];
        }
      } else { // Rarefaction wave
        PetscReal rstarR = wR[0] * PetscPowReal(p_ratioR, 1 / phys->gamma);
        PetscReal astarR = PetscSqrtReal(phys->gamma * pstar / rstarR);
        if (unR + aR < 0) { // Right of the rarefaction
          un = unR;
          p = wR[dim + 1];
          rho = wR[0];
        } else if (ustar + astarR > 0) { // Left of the rarefaction
          un = ustar;
          p = pstar;
          rho = rstarR;
        } else { // In the rarefaction fan
          un = -aR / (alpha * phys->gamma) + delta * unR;
          p = wR[dim + 1] * PetscPowReal(un / aR, 1 / beta);
          rho = wR[0] * PetscPowReal(un / aR, 1 / phys->gamma);
        }
      }
    }
  }

  PetscReal us, unorm2;
  { // Dot product u.S, speed squared norm
    us = un * area;
    unorm2 = 0;
    for (PetscInt i = 0; i < dim; i++) unorm2 += PetscSqr(un * nn[i] + ut[i]);
  }

  { // Evaluating flux
    flux[0] = rho * us;
    for (PetscInt i = 0; i < dim; i++) flux[1 + i] = rho * (un * nn[i] + ut[i]) * us + p * n[i];
    flux[dim + 1] = (p / (2 * beta) + 0.5 * rho * unorm2) * us;
  }

  PetscFunctionReturnVoid();
}


/*

*/
static void RiemannSolver_LaxFriedrichs(PetscInt dim, PetscInt Nc,
                                        const PetscReal x[], const PetscReal n[], const PetscReal uL[], const PetscReal uR[],
                                        PetscInt numConstants, const PetscScalar constants[], PetscReal flux[], void *ctx){
  Physics phys = (Physics) ctx;

  PetscFunctionBeginUser;
  SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_MAX_VALUE, "LaxFriedrichs Riemann solver not implemented yet");

  PetscReal wL[Nc], wR[Nc];
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


/*
  Adaptive Noniterative Riemann Solver (Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics")
*/
static void RiemannSolver_ANRS(PetscInt dim, PetscInt Nc,
                               const PetscReal x[], const PetscReal n[], const PetscReal uL[], const PetscReal uR[],
                               PetscInt numConstants, const PetscScalar constants[], PetscReal flux[], void *ctx){
  Physics phys = (Physics) ctx;

  PetscFunctionBeginUser;

  PetscReal beta  = (phys->gamma - 1) / (2 * phys->gamma); // = 1 - alpha
  PetscReal delta = (phys->gamma - 1) / (phys->gamma + 1); // = beta / alpha

  PetscReal area, nn[dim];
  { // n = area * nn, nn normal unitary vector to the surface
    area = 0;
    for (PetscInt i = 0; i < dim; i++) area += PetscSqr(n[i]);
    area = PetscSqrtReal(area);
    for (PetscInt i = 0; i < dim; i++) nn[i] = n[i] / area;
  }

  PetscReal wL[Nc], wR[Nc];
  { // Primitive variables (rho, u_1, ..., u_dim, p)
    ConservativeToPrimitive(phys, uL, wL);
    ConservativeToPrimitive(phys, uR, wR);
  }

  PetscReal unL, unR, utL[dim], utR[dim];
  { // Normal and tangent speeds
    unL = 0;
    unR = 0;
    for (PetscInt i = 0; i < dim; i++){
      unL += wL[1 + i] * nn[i];
      unR += wR[1 + i] * nn[i];
    }
    for (PetscInt i = 0; i < dim; i++){
      utL[i] = wL[1 + i] - unL * nn[i];
      utR[i] = wR[1 + i] - unR * nn[i];
    }
  }

  PetscReal aL, aR;
  { // Speeds of sound
    aL = PetscSqrtReal(phys->gamma * wL[dim + 1] / wL[0]);
    aR = PetscSqrtReal(phys->gamma * wR[dim + 1] / wR[0]);
  }

  PetscReal pmin, pmax;
  { // Lower and upper pressures
    if (wL[dim + 1] < wR[dim + 1]) {
      pmin = wL[dim + 1];
      pmax = wR[dim + 1];
    } else {
      pmin = wR[dim + 1];
      pmax = wL[dim + 1];
    }
  }

  PetscReal pstar, ustar, rstarL, rstarR;
  { // Solving pressure problem
    PetscReal CL = aL * wL[0];
    PetscReal CR = aR * wR[0];

    pstar = (CR * wL[dim + 1] + CL * wR[dim + 1] + CL * CR * (unL - unR)) / (CL + CR);
    ustar = (CL * unL + CR * unR + wL[dim + 1] - wR[dim + 1]) / (CL + CR);

    if (pmax / pmin < phys->riemann_ctx.q_user && pmin <= pstar && pstar <= pmax) { // PVRS
      rstarL = wL[0] + (pstar - wL[dim + 1]) / PetscSqr(aL);
      rstarR = wR[0] + (pstar - wR[dim + 1]) / PetscSqr(aR);

    } else if (pstar < pmin) { // TRRS
      PetscReal PLR = PetscPowReal(wL[dim + 1] / wR[dim + 1], beta);

      pstar = wL[dim + 1] * PetscPowReal((aL + aR - (unR - unL) * (phys->gamma - 1) / 2) / (aL + aR * PLR), 1 / beta);
      ustar = (PLR * unL / aL + unR / aR + 2 * (PLR - 1) / (phys->gamma - 1)) / (PLR / aL + 1 / aR);

      rstarL = wL[0] * PetscPowReal(pstar / wL[dim + 1], 1 / phys->gamma);
      rstarR = wR[0] * PetscPowReal(pstar / wR[dim + 1], 1 / phys->gamma);

    } else { // TSRS
      if (pstar < 0) pstar = 0;
      PetscReal gL = PetscSqrtReal(2 / (wL[0] * ((phys->gamma + 1) * pstar + (phys->gamma - 1) * wL[dim + 1])));
      PetscReal gR = PetscSqrtReal(2 / (wR[0] * ((phys->gamma + 1) * pstar + (phys->gamma - 1) * wR[dim + 1])));

      pstar = (gL * wL[dim + 1] + gR * wR[dim + 1] + unL - unR) / (gL + gR);
      ustar = (unL + unR + (pstar - wR[dim + 1]) * gR - (pstar - wL[dim + 1]) * gL) / 2;

      PetscReal p_ratioL = pstar / wL[dim + 1];
      PetscReal p_ratioR = pstar / wR[dim + 1];

      rstarL = wL[0] * (delta + p_ratioL) / (1 + delta * p_ratioL);
      rstarR = wR[0] * (delta + p_ratioR) / (1 + delta * p_ratioR);
    }
  }

  PetscReal rho, un, *ut, p;
  { // Sampling the solution : density, normal and tangent speeds, and pressure at the interface
    if (ustar > 0) { // left side of contact
      ut = utL;
      PetscReal p_ratioL = pstar / wL[dim + 1];
      if (p_ratioL > 1) { // Shock wave
        if (unL - aL * PetscSqrtReal((1 - beta) * p_ratioL + beta) < 0) { // Shock speed < 0, star state
          un = ustar;
          p = pstar;
          rho = rstarL;
        } else { // Shock speed >= 0, left state
          un = unL;
          p = wL[dim + 1];
          rho = wL[0];
        }
      } else { // Rarefaction wave
        PetscReal astarL = PetscSqrtReal(phys->gamma * pstar / rstarL);
        if (unL - aL > 0) { // Left of the rarefaction
          un = unL;
          p = wL[dim + 1];
          rho = wL[0];
        } else if (ustar - astarL < 0) { // Right of the rarefaction
          un = ustar;
          p = pstar;
          rho = rstarL;
        } else { // In the rarefaction fan
          un = aL / ((1 - beta) * phys->gamma) + delta * unL;
          p = wL[dim + 1] * PetscPowReal(un / aL, 1 / beta);
          rho = wL[0] * PetscPowReal(un / aL, 1 / phys->gamma);
        }
      }
    } else { // Right side of the contact
      ut = utR;
      PetscReal p_ratioR = pstar / wR[dim + 1];
      if (p_ratioR > 1) { // Shock wave
        if (unR + aR * PetscSqrtReal((1 - beta) * p_ratioR + beta) > 0) { // Shock speed > 0, star state
          un = ustar;
          p = pstar;
          rho = rstarR;
        } else { // Shock speed <= 0, right state
          un = unR;
          p = wR[dim + 1];
          rho = wR[0];
        }
      } else { // Rarefaction wave
        PetscReal astarR = PetscSqrtReal(phys->gamma * pstar / rstarR);
        if (unR + aR < 0) { // Right of the rarefaction
          un = unR;
          p = wR[dim + 1];
          rho = wR[0];
        } else if (ustar + astarR > 0) { // Left of the rarefaction
          un = ustar;
          p = pstar;
          rho = rstarR;
        } else { // In the rarefaction fan
          un = -aR / ((1 - beta) * phys->gamma) + delta * unR;
          p = wR[dim + 1] * PetscPowReal(un / aR, 1 / beta);
          rho = wR[0] * PetscPowReal(un / aR, 1 / phys->gamma);
        }
      }
    }
  }

  PetscReal us, unorm2;
  { // Dot product u.S, speed squared norm
    us = un * area;
    unorm2 = 0;
    for (PetscInt i = 0; i < dim; i++) unorm2 += PetscSqr(un * nn[i] + ut[i]);
  }

  { // Evaluating flux
    flux[0] = rho * us;
    for (PetscInt i = 0; i < dim; i++) flux[1 + i] = rho * (un * nn[i] + ut[i]) * us + p * n[i];
    flux[dim + 1] = (p / (2 * beta) + 0.5 * rho * unorm2) * us;
  }

  PetscFunctionReturnVoid();
}


/*
  Entropy fixes for Roe-Pike Riemann solver
  HH1 and HH2 are introduced in Harten and Hyman, "Self adjusting grid methods for one-dimensional hyperbolic conservation laws",
  but an easier explanation can be found in Pelanti et al., "A review of entropy fixes as applied to Roe’s linearization"
*/
static void RiemannSolver_RoePike_EntropyFix_None(PetscReal *a_k, PetscReal a_kL, PetscReal a_kR){PetscFunctionReturnVoid();}
static void RiemannSolver_RoePike_EntropyFix_HH1(PetscReal *a_k, PetscReal a_kL, PetscReal a_kR){
  PetscReal delta_k = PetscMax(*a_k - a_kL, a_kR - *a_k); if (PetscAbs(*a_k) < delta_k) *a_k = delta_k;
}
static void RiemannSolver_RoePike_EntropyFix_HH2(PetscReal *a_k, PetscReal a_kL, PetscReal a_kR){
  PetscReal delta_k = PetscMax(*a_k - a_kL, a_kR - *a_k); if (PetscAbs(*a_k) < delta_k) *a_k = (PetscSqr(*a_k) / delta_k + delta_k) / 2;
}

/*
  Roe-Pike approximate Riemann solver
  For details see Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics"
*/
static void RiemannSolver_RoePike(PetscInt dim, PetscInt Nc,
                               const PetscReal x[], const PetscReal n[], const PetscReal uL[], const PetscReal uR[],
                               PetscInt numConstants, const PetscScalar constants[], PetscReal flux[], void *ctx){
  Physics phys = (Physics) ctx;

  PetscFunctionBeginUser;

  PetscReal area, nn[dim];
  { // n = area * nn, nn normal unitary vector to the surface
    area = 0;
    for (PetscInt i = 0; i < dim; i++) area += PetscSqr(n[i]);
    area = PetscSqrtReal(area);
    for (PetscInt i = 0; i < dim; i++) nn[i] = n[i] / area;
  }

  PetscReal wL[Nc], wR[Nc];
  { // Primitive variables (rho, u_1, ..., u_dim, p)
    ConservativeToPrimitive(phys, uL, wL);
    ConservativeToPrimitive(phys, uR, wR);
  }

  PetscReal unL, unR, utL[dim], utR[dim];
  { // Normal and tangent speeds
    unL = 0;
    unR = 0;
    for (PetscInt i = 0; i < dim; i++){
      unL += wL[1 + i] * nn[i];
      unR += wR[1 + i] * nn[i];
    }
    for (PetscInt i = 0; i < dim; i++){
      utL[i] = wL[1 + i] - unL * nn[i];
      utR[i] = wR[1 + i] - unR * nn[i];
    }
  }

  PetscReal aL, aR;
  { // Speeds of sound
    aL = PetscSqrtReal(phys->gamma * wL[dim + 1] / wL[0]);
    aR = PetscSqrtReal(phys->gamma * wR[dim + 1] / wR[0]);
  }

  PetscReal rROE, uROE, utROE[dim], hROE, aROE, unorm2ROE;
  { // Roe averages
    PetscReal ratioLR = PetscSqrtReal(wL[0] * wR[0]);

    rROE = PetscSqrtReal(wL[0] * wR[0]);
    uROE = (ratioLR * unL + unR) / (ratioLR + 1);
    for (PetscInt i = 0; i < dim; i++) utROE[i] = (ratioLR * utL[i] + utR[i]) / (ratioLR + 1);

    unorm2ROE = 0;
    for (PetscInt i = 0; i < dim; i++) unorm2ROE += PetscSqr(uROE * nn[i] + utROE[i]);

    hROE = (uL[dim + 1] + wL[dim + 1] + ratioLR * (uR[dim + 1] + wR[dim + 1])) / (wL[0] + ratioLR);
    aROE = PetscSqrtReal((phys->gamma - 1) * (hROE - unorm2ROE/ 2));
  }

  PetscReal lambda_p, lambda_m, lambda_u;
  { // Eigenvalues
    lambda_p = uROE + aROE;
    lambda_u = uROE;
    lambda_m = uROE - aROE;
  }

  { // Entropy fix
    phys->riemann_ctx.entropy_fix(&lambda_p, unL + aL, unR + aR);
    phys->riemann_ctx.entropy_fix(&lambda_u, unL, unR);
    phys->riemann_ctx.entropy_fix(&lambda_m, unL - aL, unR - aR);
  }

  PetscReal coeff_p, coeff_m, coeff_u;
  { // Wave strengths * |eigenvalue|
    coeff_p = PetscAbs(lambda_p) * (wR[dim + 1] - wL[dim + 1] + rROE * aROE * (unR - unL)) / (2 * PetscSqr(aROE));
    coeff_u = PetscAbs(lambda_u) * (wR[0] - wL[0] - (wR[dim + 1] - wL[dim + 1]) / PetscSqr(aROE));
    coeff_m = PetscAbs(lambda_m) * (wR[dim + 1] - wL[dim + 1] - rROE * aROE * (unR - unL)) / (2 * PetscSqr(aROE));
  }

  { // Evaluating flux
    flux[0] = wL[0] * unL + wR[0] * unR;
    for (PetscInt i = 0; i < dim; i++) flux[1 + i] = uL[1 + i] * unL + uR[1 + i] * unR + (wL[dim + 1] + wR[dim + 1]) * nn[i];
    flux[dim + 1] = (uL[dim + 1] + wL[dim + 1]) * unL + (uR[dim + 1] + wR[dim + 1]) * unR;

    PetscReal coeff_rE = 0;
    for (PetscInt i = 0; i < dim; i++) coeff_rE = utROE[i] * (wR[1 + i] - wL[1 + i] - (unR - unL) * nn[i]);

    flux[0] -= coeff_p + coeff_u + coeff_m;
    for (PetscInt i = 0; i < dim; i++) flux[1 + i] -= (coeff_p + coeff_m) * (uROE * n[i] + utROE[i]) + rROE * PetscAbs(uROE) * (utR[i] - utL[i]) + (coeff_p - coeff_m) * aROE * nn[i];
    flux[dim + 1] -= coeff_m * (hROE - uROE * aROE) + coeff_u * unorm2ROE / 2 + coeff_p * (hROE + uROE * aROE) + rROE * PetscAbs(uROE) * coeff_rE;

    for (PetscInt i = 0; i < Nc; i++) flux[i] *= area / 2;
  }

  PetscFunctionReturnVoid();
}



PetscErrorCode PhysicsRiemannSetFromOptions(MPI_Comm comm,
                                            void (**riemann_solver)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void*),
                                            union RiemannCtx *riemann_ctx){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  { // Getting Riemann solver from options
    ierr = PetscOptionsBegin(comm, NULL, "Physical solver", NULL);                                                                    CHKERRQ(ierr);
    char riemann_name[256] = "exact";
    PetscFunctionList riemann_list = NULL;
    ierr = PetscFunctionListAdd(&riemann_list, "advection", RiemannSolver_AdvectionX);                                                CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&riemann_list, "exact",     RiemannSolver_Exact);                                                     CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&riemann_list, "lax",       RiemannSolver_LaxFriedrichs);                                             CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&riemann_list, "anrs",      RiemannSolver_ANRS);                                                      CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&riemann_list, "roe",       RiemannSolver_RoePike);                                                   CHKERRQ(ierr);
    ierr = PetscOptionsFList("-riemann", "Riemann Solver", "", riemann_list, riemann_name, riemann_name, sizeof(riemann_name), NULL); CHKERRQ(ierr);
    ierr = PetscFunctionListFind(riemann_list, riemann_name, riemann_solver);                                                         CHKERRQ(ierr);
    if (!*riemann_solver) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Unknown Riemann solver: '%s'", riemann_name);
    ierr = PetscFunctionListDestroy(&riemann_list);                                                                                   CHKERRQ(ierr);
    ierr = PetscOptionsEnd();                                                                                                         CHKERRQ(ierr);
  }

  if (*riemann_solver == RiemannSolver_AdvectionX) {
    ierr = PetscOptionsBegin(comm, "", "Options for the Advection Riemann solver", NULL); CHKERRQ(ierr);
    riemann_ctx->advection_speed = 1;
    ierr = PetscOptionsReal("-riemann_advection_speed", "Advection speed", NULL, riemann_ctx->advection_speed, &riemann_ctx->advection_speed, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  } else if (*riemann_solver == RiemannSolver_Exact) {
    ierr = PetscOptionsBegin(comm, "", "Options for the Exact Riemann solver", NULL); CHKERRQ(ierr);

    char p_solver_name[256] = "newton";
    PetscFunctionList p_solver_list = NULL;
    ierr = PetscFunctionListAdd(&p_solver_list, "newton", RiemannSolver_Exact_PressureSolver_Newton); CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&p_solver_list, "fp",     RiemannSolver_Exact_PressureSolver_FP);     CHKERRQ(ierr);
    ierr = PetscOptionsFList("-riemann_exact_p_solver", "Riemann Exact Pressure Solver", "", p_solver_list, p_solver_name, p_solver_name, sizeof(p_solver_name), NULL); CHKERRQ(ierr);
    ierr = PetscFunctionListFind(p_solver_list, p_solver_name, &riemann_ctx->pressure_solver); CHKERRQ(ierr);
    if (!riemann_ctx->pressure_solver) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Unknown Pressure solver: '%s'", p_solver_name);
    ierr = PetscFunctionListDestroy(&p_solver_list); CHKERRQ(ierr);

    riemann_ctx->pressure_solver_rtol = 1E-6;
    riemann_ctx->pressure_solver_niter = 20;
    riemann_ctx->pressure_solver_eps = 1E-5;
    ierr = PetscOptionsReal("-riemann_exact_p_solver_rtol", "Relative tolerance for the pressure solver", NULL, riemann_ctx->pressure_solver_rtol, &riemann_ctx->pressure_solver_rtol, NULL);                                     CHKERRQ(ierr);
    ierr = PetscOptionsReal("-riemann_exact_p_solver_niter", "Maximum number of iterations for the pressure solver", NULL, riemann_ctx->pressure_solver_niter, &riemann_ctx->pressure_solver_niter, NULL);                        CHKERRQ(ierr);
    ierr = PetscOptionsReal("-riemann_exact_p_solver_eps", "Epsilon on the pressure solver, triggers linearisation of the rarefaction formula", NULL, riemann_ctx->pressure_solver_eps, &riemann_ctx->pressure_solver_eps, NULL); CHKERRQ(ierr);

    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  } else if (*riemann_solver == RiemannSolver_ANRS) {
    ierr = PetscOptionsBegin(comm, "", "Options for the Adaptive Noniterative Riemann Solver", NULL); CHKERRQ(ierr);
    riemann_ctx->q_user = 2;
    ierr = PetscOptionsReal("-riemann_anrs_q", "Pressure ratio over which the PVRS is not used", NULL, riemann_ctx->q_user, &riemann_ctx->q_user, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  } else if (*riemann_solver == RiemannSolver_RoePike) {
    ierr = PetscOptionsBegin(comm, "", "Options for the Roe-Pike Riemann Solver", NULL); CHKERRQ(ierr);
    char entropy_fix_name[256] = "none";
    PetscFunctionList entropy_fix_list = NULL;
    ierr = PetscFunctionListAdd(&entropy_fix_list, "none", RiemannSolver_RoePike_EntropyFix_None); CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&entropy_fix_list, "hh1",  RiemannSolver_RoePike_EntropyFix_HH1);  CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&entropy_fix_list, "hh2",  RiemannSolver_RoePike_EntropyFix_HH2);  CHKERRQ(ierr);
    ierr = PetscOptionsFList("-riemann_roe_entropy_fix", "Roe-Pike solver entropy fix", "", entropy_fix_list, entropy_fix_name, entropy_fix_name, sizeof(entropy_fix_name), NULL); CHKERRQ(ierr);
    ierr = PetscFunctionListFind(entropy_fix_list, entropy_fix_name, &riemann_ctx->entropy_fix); CHKERRQ(ierr);
    if (!riemann_ctx->entropy_fix) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Unknown Entropy fix: '%s'", entropy_fix_name);
    ierr = PetscFunctionListDestroy(&entropy_fix_list); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

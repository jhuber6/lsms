/* -*- mode: C++; c-file-style: "bsd"; c-basic-offset: 2; indent-tabs-mode: nil * -*- */

#include "Complex.hpp"
#include "Matrix.hpp"
#include <cmath>
#include <vector>

#include "Main/LSMSMode.hpp"
#include "Misc/Coeficients.hpp"
#include "Misc/Indices.hpp"
#include "Misc/associatedLegendreFunction.hpp"
#include "MultipleScattering.hpp"
#include "SingleSite/SingleSiteScattering.hpp"

// we might want to distinguish between systems where all lmax (and consequently
// kkrsz_ns) are the same and systems with potential different lmax on different
// atoms and l steps

inline static void calculateHankel(Complex prel, Real r, int lend,
                                   Complex *hfn, Complex *ilp1) {
  const Complex sqrtm1(0.0, 1.0);
  Complex z = prel * r;
  hfn[0] = -sqrtm1;
  hfn[1] = -1.0 - sqrtm1 / z;
  for (int l = 1; l < lend; l++) {
    hfn[l + 1] = (2.0 * l + 1.0) * hfn[l] / z - hfn[l - 1];
  }
  /*
c             l+1
c     hfn = -i   *h (k*R  )*sqrt(E)
c                  l    ij
*/
  z = std::exp(sqrtm1 * z) / r;
  for (int l = 0; l <= lend; l++) {
    hfn[l] = -hfn[l] * z * ilp1[l];
  }
}

inline void calculateSinCosPowers(Real *rij, int lend, Real *sinmp,
                                  Real *cosmp) {
  const Real ptol = 1.0e-6;
  Real pmag = std::sqrt(rij[0] * rij[0] + rij[1] * rij[1]);
  cosmp[0] = 1.0;
  sinmp[0] = 0.0;
  if (pmag > ptol) {
    cosmp[1] = rij[0] / pmag;
    sinmp[1] = rij[1] / pmag;
  } else {
    cosmp[1] = 0.0;
    sinmp[1] = 0.0;
  }
  for (int m = 2; m <= lend; m++) {
    cosmp[m] = cosmp[m - 1] * cosmp[1] - sinmp[m - 1] * sinmp[1];
    sinmp[m] = sinmp[m - 1] * cosmp[1] + cosmp[m - 1] * sinmp[1];
  }
}

Complex dlmFunction(Complex *hfn, double *cosmp, double *sinmp, double *plm,
                    int l, int m) {
  int mAbs = std::abs(m);

  Complex dlm = hfn[l] * plm[(l * (l + 1) / 2 + mAbs)];
  if (m == 0)
    return dlm;

  if (m < 0) {
    dlm = dlm * Complex(cosmp[mAbs], sinmp[mAbs]);
    if (mAbs & 0x01) // m is odd
      dlm = -dlm;
  } else {
    dlm = dlm * Complex(cosmp[mAbs], -sinmp[mAbs]);
  }

  return dlm;
}

void setBGijOMP(int ir1, int ir2, int iOffset, int jOffset, int kkrsz,
                int n_spin_cant, int *LIZlmax, Complex *bgij, int nrmat_ns) {
  if (n_spin_cant == 1)
    return;

  int kkri = (LIZlmax[ir1] + 1) * (LIZlmax[ir1] + 1);
  int kkrj = (LIZlmax[ir2] + 1) * (LIZlmax[ir2] + 1);

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < kkri; i++) {
    for (int j = 0; j < kkrj; j++) {
      bgij[(jOffset + j)        * nrmat_ns + iOffset + kkri + i] = 0.0;
      bgij[(jOffset + kkrj + j) * nrmat_ns + iOffset +        i] = 0.0;
      bgij[(jOffset + kkrj + j) * nrmat_ns + iOffset + kkri + i] = bgij[(jOffset + j) * nrmat_ns + iOffset + i];
    }
  }
}

// Allocate shared memory
// Map arrays on the device
void buildBGijOMP(int ir1, int ir2, Real *rij, Complex energy, Complex prel,
                  int iOffset, int jOffset, int ndlm, int ndlj, int maxlmax,
                  int *LIZlmax, int *lofk, int *mofk, Complex *illp,
                  int ndlj_illp, Complex* ilp1, Real *cgnt, int lmaxp1_cgnt, int ndlj_cgnt,
                  Complex *bgij, int nrmat_ns) {
  Complex hfn[2 * maxlmax + 1];
  Real sinmp[2 * maxlmax + 1];
  Real cosmp[2 * maxlmax + 1];
  Real plm[ndlm];
  Complex dlm[ndlj];
  Real r = std::sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]);
  int lmax1 = LIZlmax[ir1];
  int lmax2 = LIZlmax[ir2];
  int kkri = (lmax1 + 1) * (lmax1 + 1);
  int kkrj = (lmax2 + 1) * (lmax2 + 1);
  int lend = lmax1 + lmax2;

  Real pi4 = 4.0 * 2.0 * std::asin(1.0);
  Real cosTheta = rij[2] / r;

  calculateHankel(prel, r, lend, hfn, ilp1);

  associatedLegendreFunctionNormalized<Real>(cosTheta, lend, plm);
  // for associatedLegendreFunctionNormalized all clm[i] == 1.0
  //     calculate cos(phi) and sin(phi) .................................
  // needs to be serial
  calculateSinCosPowers(rij, lend, sinmp, cosmp);

  // #pragma omp parallel
  // {
  // can be parallel
  //
  //     ================================================================
  //     calculate g(R_ij)...............................................
  for (int i = 0; i < kkri; i++)
    for (int j = 0; j < kkrj; j++)
      bgij[(jOffset + j) * nrmat_ns + iOffset + i] = 0.0;

  //     loop over l1,m1............................................
  //        loop over l2,m2..............................................
  for (int lm1 = 0; lm1 < kkrj; lm1++) {
    for (int lm2 = 0; lm2 < kkri; lm2++) {
      int l1 = lofk[lm1];
      int m1 = mofk[lm1];

      int l2 = lofk[lm2];
      int m2 = mofk[lm2];
      /*
         ==========================================================
         l2-l1
         illp(lm2,lm1) = i

         perform sum over l3 with gaunt # ......................
         ==========================================================
         */
      int m3 = m2 - m1;
      int llow = std::max(std::abs(m3), std::abs(l1 - l2));
      if (std::abs(prel) == 0.0)
        llow = l1 + l2;
      for (int l3 = l1 + l2; l3 >= llow; l3 -= 2) {
        bgij[(jOffset + lm1) * nrmat_ns + iOffset + lm2] +=
            cgnt[lm2 * lmaxp1_cgnt * ndlj_cgnt + lm1 * lmaxp1_cgnt + l3 / 2] *
            dlmFunction(hfn, cosmp, sinmp, plm, l3, m3);
      }
      bgij[(jOffset + lm1) * nrmat_ns + iOffset + lm2] *=
          pi4 * illp[lm1 * ndlj_illp + lm2];
    }
  }
  // } // End of parallel
}

// Remove zgemm calls
// Replace dlm LUT with function -- Too large for shared memory
void buildKKRMatrixLMaxIdenticalOMP(LSMSSystemParameters &lsms,
                                    LocalTypeInfo &local, AtomData &atom,
                                    int iie, Complex energy, Complex prel,
                                    Matrix<Complex> &m) {
  int nrmat_ns = lsms.n_spin_cant * atom.nrmat; // total size of the kkr matrix
  int kkrsz_ns = lsms.n_spin_cant * atom.kkrsz; // size of t00 block

  Complex cmone = Complex(-1.0, 0.0);
  Complex czero = 0.0;

  Matrix<Complex> bgij(nrmat_ns, nrmat_ns);

  for (int i = 0; i < nrmat_ns; i++)
    for (int j = 0; j < nrmat_ns; j++)
      m[j * nrmat_ns + i] = (i == j) ? 1.0 : 0.0;

  for (int i = 0; i < nrmat_ns; i++)
    for (int j = 0; j < nrmat_ns; j++)
      bgij[j * nrmat_ns +i] = 0.0;

  // loop over the LIZ blocks
  Complex *mDev = &m(0, 0);
  Complex *bgijDev = &bgij(0, 0);
  Complex *tmatDev = &local.tmatStore(iie * local.blkSizeTmatStore, 0);
  int tmatDim = local.tmatStore.l_dim();

  int numLIZ = atom.numLIZ;
  int kkrsz = atom.kkrsz;

  int n_spin_cant = lsms.n_spin_cant;
  int maxlmax = lsms.maxlmax;
  int ndlm = lsms.angularMomentumIndices.ndlm;
  int ndlj = lsms.angularMomentumIndices.ndlj;

  Complex *ilp1 = IFactors::ilp1.data();

  Complex *illp = &IFactors::illp(0, 0);
  int ndlj_illp = IFactors::illp.l_dim();

  Real *cgnt = &GauntCoeficients::cgnt(0, 0, 0);
  int lmaxp1_cgnt = GauntCoeficients::cgnt.l_dim1();
  int ndlj_cgnt = GauntCoeficients::cgnt.l_dim2();

  int *LIZlmax = atom.LIZlmax.data();
  int *LIZStoreIdx = atom.LIZStoreIdx.data();
  Real *LIZPos = &atom.LIZPos(0, 0);

  int *lofk = AngularMomentumIndices::lofk.data(); // [0:ndlj]
  int *mofk = AngularMomentumIndices::mofk.data(); // [0:ndlm]

//#pragma omp target data \
//  map(to: illp[0 : 
//  map(to: lofk[0 : ndlj]) \
//  map(to: mofk[0 : ndlm])
  for (int ir1 = 0; ir1 < numLIZ; ir1++) {
    for (int ir2 = 0; ir2 < numLIZ; ir2++) {
      if (ir1 != ir2) {
        Real rij[3];

        int iOffset = ir1 * kkrsz_ns;
        int jOffset = ir2 * kkrsz_ns;

        int lmax1 = LIZlmax[ir1];
        int lmax2 = LIZlmax[ir2];

        int kkr1 = (lmax1 + 1) * (lmax1 + 1);
        int kkr2 = (lmax2 + 1) * (lmax2 + 1);

        int kkr1_ns = kkr1 * n_spin_cant;
        int kkr2_ns = kkr2 * n_spin_cant;

        rij[0] = LIZPos[ir1 * 3 + 0] - LIZPos[ir2 * 3 + 0];
        rij[1] = LIZPos[ir1 * 3 + 1] - LIZPos[ir2 * 3 + 1];
        rij[2] = LIZPos[ir1 * 3 + 2] - LIZPos[ir2 * 3 + 2];

        buildBGijOMP(ir1, ir2, rij, energy, prel, iOffset, jOffset, ndlm, ndlj,
                     maxlmax, LIZlmax, lofk, mofk, illp, ndlj_illp, ilp1, cgnt,
                     lmaxp1_cgnt, ndlj_cgnt, bgijDev, nrmat_ns);

        setBGijOMP(ir1, ir2, iOffset, jOffset, kkrsz, n_spin_cant, LIZlmax,
                   bgijDev, nrmat_ns);

        for (int i = 0; i < kkr1_ns; i++) {
          for (int j = 0; j < kkr2_ns; j++) {
            Complex sum = 0.0;
            for (int k = 0; k < kkr1_ns; k++)
              sum += cmone *
                     tmatDev[LIZStoreIdx[ir1] * tmatDim + k * kkr1_ns + i] *
                     bgijDev[(jOffset + j) * nrmat_ns + iOffset + k];

            mDev[(jOffset + j) * nrmat_ns + iOffset + i] = sum;
          }
        }
      }
    }
  }
}

void buildKKRMatrixLMaxDifferentOMP(LSMSSystemParameters &lsms,
                                    LocalTypeInfo &local, AtomData &atom,
                                    int iie, Complex energy, Complex prel,
                                    Matrix<Complex> &m) {
  int nrmat_ns = lsms.n_spin_cant * atom.nrmat; // total size of the kkr matrix
  int kkrsz_ns = lsms.n_spin_cant * atom.kkrsz; // size of t00 block

  const Complex cmone = Complex(-1.0, 0.0);
  const Complex czero = 0.0;

  Matrix<Complex> bgij(nrmat_ns, nrmat_ns);

  for (int i = 0; i < nrmat_ns; i++)
    for (int j = 0; j < nrmat_ns; j++)
      m[j * nrmat_ns + i] = (i == j) ? 1.0 : 0.0;

  for (int i = 0; i < nrmat_ns; i++)
    for (int j = 0; j < nrmat_ns; j++)
      bgij[j * nrmat_ns +i] = 0.0;

  std::vector<int> offsets(atom.numLIZ);
  offsets[0] = 0;
  for (int ir = 1; ir < atom.numLIZ; ir++)
    offsets[ir] = offsets[ir - 1] + lsms.n_spin_cant *
                                        (atom.LIZlmax[ir - 1] + 1) *
                                        (atom.LIZlmax[ir - 1] + 1);

  Complex *mDev = &m(0, 0);
  Complex *bgijDev = &bgij(0, 0);
  Complex *tmatDev = &local.tmatStore(iie * local.blkSizeTmatStore, 0);
  int tmatDim = local.tmatStore.l_dim();

  int numLIZ = atom.numLIZ;
  int kkrsz = atom.kkrsz;

  int n_spin_cant = lsms.n_spin_cant;
  int maxlmax = lsms.maxlmax;
  int ndlm = lsms.angularMomentumIndices.ndlm;
  int ndlj = lsms.angularMomentumIndices.ndlj;

  Complex *ilp1 = IFactors::ilp1.data();

  Complex *illp = &IFactors::illp(0, 0);
  int ndlj_illp = IFactors::illp.l_dim();

  Real *cgnt = &GauntCoeficients::cgnt(0, 0, 0);
  int lmaxp1_cgnt = GauntCoeficients::cgnt.l_dim1();
  int ndlj_cgnt = GauntCoeficients::cgnt.l_dim2();

  int *LIZlmax = atom.LIZlmax.data();
  int *LIZStoreIdx = atom.LIZStoreIdx.data();
  Real *LIZPos = &atom.LIZPos(0, 0);

  int *lofk = AngularMomentumIndices::lofk.data();
  int *mofk = AngularMomentumIndices::mofk.data();

  int *offsetsDev = offsets.data();

  // loop over the LIZ blocks
  for (int ir1 = 0; ir1 < numLIZ; ir1++) {
    for (int ir2 = 0; ir2 < numLIZ; ir2++) {
      if (ir1 != ir2) {
        Real rij[3];

        int iOffset = offsetsDev[ir1];
        int jOffset = offsetsDev[ir2];

        int lmax1 = LIZlmax[ir1];
        int lmax2 = LIZlmax[ir2];

        int kkr1 = (lmax1 + 1) * (lmax1 + 1);
        int kkr2 = (lmax2 + 1) * (lmax2 + 1);

        int kkr1_ns = kkr1 * n_spin_cant;
        int kkr2_ns = kkr2 * n_spin_cant;

        rij[0] = LIZPos[ir1 * 3 + 0] - LIZPos[ir2 * 3 + 0];
        rij[1] = LIZPos[ir1 * 3 + 1] - LIZPos[ir2 * 3 + 1];
        rij[2] = LIZPos[ir1 * 3 + 2] - LIZPos[ir2 * 3 + 2];

        buildBGijOMP(ir1, ir2, rij, energy, prel, iOffset, jOffset, ndlm, ndlj,
                     maxlmax, LIZlmax, lofk, mofk, illp, ndlj_illp, ilp1, cgnt,
                     lmaxp1_cgnt, ndlj_cgnt, bgijDev, nrmat_ns);

        setBGijOMP(ir1, ir2, iOffset, jOffset, kkrsz, n_spin_cant, LIZlmax,
                   bgijDev, nrmat_ns);

        for (int i = 0; i < kkr1_ns; i++) {
          for (int j = 0; j < kkr2_ns; j++) {
            Complex sum = 0.0;
            for (int k = 0; k < kkr1_ns; k++)
              sum += cmone *
                     tmatDev[LIZStoreIdx[ir1] * tmatDim + k * kkr1_ns + i] *
                     bgijDev[(jOffset + j) * nrmat_ns + iOffset + k];

            mDev[(jOffset + j) * nrmat_ns + iOffset + i] = sum;
          }
        }
      }
    }
  }
}

// *** Check output between each stage to make sure the changes are correct
// 1. Rewrite ZGEMM to run using loops inside the device !
// 2. Rewrite DLM LUT to use a function !
// 3. Add OpenMP with no parallelism and get the data onto the device (M on
// device) ?
// 4. Add CUDA block parallelism across the blocks and threads inside the
// SetBjij / buildBGij routines
// 5. Keep M on the device and set up function calls to call an OpenMP LU solver
// 6. Write LU solver using FLENS with Offloading (Slow but looking for
// functionality)
// 7. Optimize, check performance
void buildKKRMatrixOMP(LSMSSystemParameters &lsms, LocalTypeInfo &local,
                       AtomData &atom, int iie, Complex energy, Complex prel,
                       Matrix<Complex> &m) {
  // decide between identical lmax and different lmax:

  bool lmaxIdentical = true;

  if (atom.LIZlmax[0] != lsms.maxlmax) {
    lmaxIdentical = false;
    printf("atom.LIZlmax[0] (=%d) != lsms.maxlmax (=%d)\n", atom.LIZlmax[0],
           lsms.maxlmax);
  }
  for (int ir = 0; ir < atom.numLIZ; ir++) {
    if (atom.LIZlmax[ir] != atom.LIZlmax[0])
      lmaxIdentical = false;
  }

  if (lmaxIdentical) {
    // printf("lmax identical in buildKKRMatrix\n");
    buildKKRMatrixLMaxIdenticalOMP(lsms, local, atom, iie, energy, prel, m);
  } else {
    // printf("lmax not identical in buildKKRMatrix\n");
    buildKKRMatrixLMaxDifferentOMP(lsms, local, atom, iie, energy, prel, m);
  }
}

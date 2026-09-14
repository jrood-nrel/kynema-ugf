// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef EIGENDECOMPOSITION_H
#define EIGENDECOMPOSITION_H

#include <FieldTypeDef.h>
#include <SimdInterface.h>
#include <KynemaUGFEnv.h>
#include <cmath>
#include <limits>
#include <type_traits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sierra {
namespace kynema_ugf {

class Realm;
namespace EigenDecomposition {

//--------------------------------------------------------------------------
//-------- symmetric diagonalize (2D) --------------------------------------
//--------------------------------------------------------------------------
template <class T>
KOKKOS_FUNCTION void
sym_diagonalize(const T (&A)[2][2], T (&Q)[2][2], T (&D)[2][2])
{
  // Note that A must be symmetric here
  const T trace = A[0][0] + A[1][1];
  const T det = A[0][0] * A[1][1] - A[0][1] * A[1][0];

  // calculate eigenvalues
  D[0][0] = stk::math::if_then_else(
    A[1][0] == 0.0, A[0][0],
    trace / 2.0 + stk::math::sqrt(trace * trace / 4.0 - det));
  D[1][1] = stk::math::if_then_else(
    A[1][0] == 0.0, A[1][1],
    trace / 2.0 - stk::math::sqrt(trace * trace / 4.0 - det));
  D[0][1] = 0.0;
  D[1][0] = 0.0;

  // calculate first eigenvector
  Q[0][0] = -A[0][1];
  Q[1][0] = A[0][0] - D[0][0];

  T norm = stk::math::sqrt(Q[0][0] * Q[0][0] + Q[1][0] * Q[1][0]);
  Q[0][0] = Q[0][0] / norm;
  Q[1][0] = Q[1][0] / norm;

  // calculate second eigenvector
  Q[0][1] = -A[1][1] + D[1][1];
  Q[1][1] = A[1][0];

  norm = stk::math::sqrt(Q[0][1] * Q[0][1] + Q[1][1] * Q[1][1]);
  Q[0][1] = Q[0][1] / norm;
  Q[1][1] = Q[1][1] / norm;

  // special case when off diagonal entries were 0, we already had a diagonal
  // matrix
  Q[0][0] = stk::math::if_then_else(A[1][0] == 0.0, 1.0, Q[0][0]);
  Q[0][1] = stk::math::if_then_else(A[1][0] == 0.0, 0.0, Q[0][1]);
  Q[1][0] = stk::math::if_then_else(A[1][0] == 0.0, 0.0, Q[1][0]);
  Q[1][1] = stk::math::if_then_else(A[1][0] == 0.0, 1.0, Q[1][1]);
}

//--------------------------------------------------------------------------
//-------- matrix_matrix_multiply 2D ---------------------------------------
//--------------------------------------------------------------------------
template <class T>
void
matrix_matrix_multiply(const T (&A)[2][2], const T (&B)[2][2], T (&C)[2][2])
{
  // C = A*B
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      T sum = 0;
      for (int k = 0; k < 2; ++k) {
        sum = sum + A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

//--------------------------------------------------------------------------
//-------- reconstruct_matrix_from_decomposition 2D ------------------------
//--------------------------------------------------------------------------
template <class T>
void
reconstruct_matrix_from_decomposition(
  const T (&D)[2][2], const T (&Q)[2][2], T (&A)[2][2])
{
  // A = Q*D*QT
  T QT[2][2];
  T B[2][2];

  // compute QT
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      QT[j][i] = Q[i][j];
    }
  }
  // mat-vec, B = Q*D
  matrix_matrix_multiply(Q, D, B);

  // mat-vec, A = (Q*D)*QT = B*QT
  matrix_matrix_multiply(B, QT, A);
}
//--------------------------------------------------------------------------
//-------- symmetric diagonalize (3D) --------------------------------------
//--------------------------------------------------------------------------
template <class T>
KOKKOS_FUNCTION void
sym_diagonalize(const T (&A)[3][3], T (&Q)[3][3], T (&D)[3][3])
{
  /*
    obtained from:
    http://stackoverflow.com/questions/4372224/
    fast-method-for-computing-3x3-symmetric-matrix-spectral-decomposition

    A must be a symmetric matrix.
    returns Q and D such that
    Diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
  */

  const int maxsteps = 24;
  T o[3], m[3];
  T q[4] = {0.0, 0.0, 0.0, 1.0};
  T jr[4];
  T sqw, sqx, sqy, sqz;
  T tmp1, tmp2, mq;
  T AQ[3][3];
  T thet, sgn, t, c;

  T jrL, oLarge, dDiff;

  for (int i = 0; i < maxsteps; ++i) {
    // quat to matrix
    sqx = q[0] * q[0];
    sqy = q[1] * q[1];
    sqz = q[2] * q[2];
    sqw = q[3] * q[3];
    Q[0][0] = (sqx - sqy - sqz + sqw);
    Q[1][1] = (-sqx + sqy - sqz + sqw);
    Q[2][2] = (-sqx - sqy + sqz + sqw);
    tmp1 = q[0] * q[1];
    tmp2 = q[2] * q[3];
    Q[1][0] = 2.0 * (tmp1 + tmp2);
    Q[0][1] = 2.0 * (tmp1 - tmp2);
    tmp1 = q[0] * q[2];
    tmp2 = q[1] * q[3];
    Q[2][0] = 2.0 * (tmp1 - tmp2);
    Q[0][2] = 2.0 * (tmp1 + tmp2);
    tmp1 = q[1] * q[2];
    tmp2 = q[0] * q[3];
    Q[2][1] = 2.0 * (tmp1 + tmp2);
    Q[1][2] = 2.0 * (tmp1 - tmp2);

    // AQ = A * Q
    AQ[0][0] = Q[0][0] * A[0][0] + Q[1][0] * A[0][1] + Q[2][0] * A[0][2];
    AQ[0][1] = Q[0][1] * A[0][0] + Q[1][1] * A[0][1] + Q[2][1] * A[0][2];
    AQ[0][2] = Q[0][2] * A[0][0] + Q[1][2] * A[0][1] + Q[2][2] * A[0][2];
    AQ[1][0] = Q[0][0] * A[0][1] + Q[1][0] * A[1][1] + Q[2][0] * A[1][2];
    AQ[1][1] = Q[0][1] * A[0][1] + Q[1][1] * A[1][1] + Q[2][1] * A[1][2];
    AQ[1][2] = Q[0][2] * A[0][1] + Q[1][2] * A[1][1] + Q[2][2] * A[1][2];
    AQ[2][0] = Q[0][0] * A[0][2] + Q[1][0] * A[1][2] + Q[2][0] * A[2][2];
    AQ[2][1] = Q[0][1] * A[0][2] + Q[1][1] * A[1][2] + Q[2][1] * A[2][2];
    AQ[2][2] = Q[0][2] * A[0][2] + Q[1][2] * A[1][2] + Q[2][2] * A[2][2];
    // D = Qt * AQ
    D[0][0] = AQ[0][0] * Q[0][0] + AQ[1][0] * Q[1][0] + AQ[2][0] * Q[2][0];
    D[0][1] = AQ[0][0] * Q[0][1] + AQ[1][0] * Q[1][1] + AQ[2][0] * Q[2][1];
    D[0][2] = AQ[0][0] * Q[0][2] + AQ[1][0] * Q[1][2] + AQ[2][0] * Q[2][2];
    D[1][0] = AQ[0][1] * Q[0][0] + AQ[1][1] * Q[1][0] + AQ[2][1] * Q[2][0];
    D[1][1] = AQ[0][1] * Q[0][1] + AQ[1][1] * Q[1][1] + AQ[2][1] * Q[2][1];
    D[1][2] = AQ[0][1] * Q[0][2] + AQ[1][1] * Q[1][2] + AQ[2][1] * Q[2][2];
    D[2][0] = AQ[0][2] * Q[0][0] + AQ[1][2] * Q[1][0] + AQ[2][2] * Q[2][0];
    D[2][1] = AQ[0][2] * Q[0][1] + AQ[1][2] * Q[1][1] + AQ[2][2] * Q[2][1];
    D[2][2] = AQ[0][2] * Q[0][2] + AQ[1][2] * Q[1][2] + AQ[2][2] * Q[2][2];
    o[0] = D[1][2];
    o[1] = D[0][2];
    o[2] = D[0][1];
    m[0] = stk::math::abs(o[0]);
    m[1] = stk::math::abs(o[1]);
    m[2] = stk::math::abs(o[2]);

    // index of largest element of offdiag
    oLarge =
      stk::math::if_then_else((m[0] > m[1]) && (m[0] > m[2]), D[1][2], 0.0);
    oLarge =
      stk::math::if_then_else((m[1] > m[2]) && (m[1] > m[0]), D[0][2], oLarge);
    oLarge =
      stk::math::if_then_else((m[2] > m[1]) && (m[2] > m[0]), D[0][1], oLarge);

    dDiff = stk::math::if_then_else(
      (m[0] > m[1]) && (m[0] > m[2]), D[2][2] - D[1][1], 0.0);
    dDiff = stk::math::if_then_else(
      (m[1] > m[2]) && (m[1] > m[0]), D[0][0] - D[2][2], dDiff);
    dDiff = stk::math::if_then_else(
      (m[2] > m[1]) && (m[2] > m[0]), D[1][1] - D[0][0], dDiff);

    // if oLarge == 0.0, then we are already diagonal
    // we need to be able to divide by thet, so set to 1.0 temporarily and
    // catch c at the end and correct it to 1 to handle the diagonal case
    const T oLargeSafe =
      stk::math::if_then_else(oLarge == T(0.0), T(1.0), oLarge);
    thet =
      stk::math::if_then_else(oLarge == 0.0, 1.0, (dDiff) / (2.0 * oLargeSafe));
    sgn = stk::math::if_then_else(thet > 0.0, 1.0, -1.0);
    thet = thet * sgn;
    // sign(T)/(|T|+sqrt(T^2+1))
    const T thetSafe = stk::math::if_then_else(thet == T(0.0), T(1.0), thet);
    t = stk::math::if_then_else(
      thet < 1.E6, sgn / (thet + stk::math::sqrt(thet * thet + 1.0)),
      0.5 * sgn / thetSafe);
    c = stk::math::if_then_else(
      oLarge == 0.0, 1.0, 1.0 / stk::math::sqrt(t * t + 1.0));

    // using 1/2 angle identity sin(a/2) = std::sqrt((1-cos(a))/2)
    // -1.0 since our quat-to-matrix convention was for v*M instead of M*v
    jr[0] = stk::math::if_then_else(
      (m[0] > m[1]) && (m[0] > m[2]),
      -1.0 * (sgn * stk::math::sqrt((1.0 - c) / 2.0)), 0.0);
    jr[1] = stk::math::if_then_else(
      (m[1] > m[2]) && (m[1] > m[0]),
      -1.0 * (sgn * stk::math::sqrt((1.0 - c) / 2.0)), 0.0);
    jr[2] = stk::math::if_then_else(
      (m[2] > m[1]) && (m[2] > m[0]),
      -1.0 * (sgn * stk::math::sqrt((1.0 - c) / 2.0)), 0.0);

    jrL = stk::math::if_then_else((m[0] > m[1]) && (m[0] > m[2]), jr[0], 0.0);
    jrL = stk::math::if_then_else((m[1] > m[2]) && (m[1] > m[0]), jr[1], jrL);
    jrL = stk::math::if_then_else((m[2] > m[1]) && (m[2] > m[0]), jr[2], jrL);

    jr[3] = stk::math::sqrt(1.0f - jrL * jrL);

    const auto check_one = jr[3] == 1.0;
    const bool exit_now = stk::simd::are_all(check_one);
    if (exit_now) {
      break; // reached limits of floating point precision
    }

    q[0] = (q[3] * jr[0] + q[0] * jr[3] + q[1] * jr[2] - q[2] * jr[1]);
    q[1] = (q[3] * jr[1] - q[0] * jr[2] + q[1] * jr[3] + q[2] * jr[0]);
    q[2] = (q[3] * jr[2] + q[0] * jr[1] - q[1] * jr[0] + q[2] * jr[3]);
    q[3] = (q[3] * jr[3] - q[0] * jr[0] - q[1] * jr[1] - q[2] * jr[2]);
    mq = stk::math::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    q[0] /= mq;
    q[1] /= mq;
    q[2] /= mq;
    q[3] /= mq;
  }
}

//--------------------------------------------------------------------------
//-------- matrix_matrix_multiply 3D ---------------------------------------
//--------------------------------------------------------------------------
template <class T>
void
matrix_matrix_multiply(const T (&A)[3][3], const T (&B)[3][3], T (&C)[3][3])
{
  // C = A*B
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      T sum = 0;
      for (int k = 0; k < 3; ++k) {
        sum = sum + A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

//--------------------------------------------------------------------------
//-------- reconstruct_matrix_from_decomposition 3D ------------------------
//--------------------------------------------------------------------------
template <class T>
void
reconstruct_matrix_from_decomposition(
  const T (&D)[3][3], const T (&Q)[3][3], T (&A)[3][3])
{
  // A = Q*D*QT
  T QT[3][3];
  T B[3][3];

  // compute QT
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      QT[j][i] = Q[i][j];
    }
  }
  // mat-vec, B = Q*D
  matrix_matrix_multiply(Q, D, B);

  // mat-vec, A = (Q*D)*QT = B*QT
  matrix_matrix_multiply(B, QT, A);
}

//--------------------------------------------------------------------------
//------------------ unsym_matrix_force_sym_3D -----------------------------
//--------------------------------------------------------------------------
template <class T>
KOKKOS_FUNCTION void
unsym_matrix_force_sym(T (&A)[3][3], T (&Q)[3][3], T (&D)[3][3])
{

  // force symmetry force
  for (int i = 0; i < 3; i++) {
    for (int j = i; j < 3; j++) {
      A[i][j] = (A[i][j] + A[j][i]) / 2.0;
      A[j][i] = A[i][j];
    }
  }

  // then call symmetric diagonalize
  sym_diagonalize(A, Q, D);
}

//--------------------------------------------------------------------------
//------------------ general_eigenvalues_3D -----------------------------
//--------------------------------------------------------------------------
template <class T>
KOKKOS_FUNCTION void
general_eigenvalues(T (&A)[3][3], T (&Q)[3][3], T (&D)[3][3])
{
  const T pi = T(M_PI);

  // Zero out Q since this only returns eigenvalues
  Q[0][0] = Q[0][1] = Q[0][2] = Q[1][0] = Q[1][1] = Q[1][2] = Q[2][0] =
    Q[2][1] = Q[2][2] = 0.0;

  // Scale the matrix so invariant-based cubic coefficients are O(1)
  T maxAbs = T(0.0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      maxAbs = stk::math::max(maxAbs, stk::math::abs(A[i][j]));
    }
  }

  T scale = maxAbs;
  if constexpr (std::is_floating_point_v<T>) {
    T minAbsNonzero = T(std::numeric_limits<double>::max());
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const T absA = stk::math::abs(A[i][j]);
        if (absA > T(0.0)) {
          minAbsNonzero = stk::math::min(minAbsNonzero, absA);
        }
      }
    }

    if ((maxAbs > T(0.0)) &&
        (minAbsNonzero < T(std::numeric_limits<double>::max()))) {
      const T maxAmp =
        stk::math::sqrt(T(std::numeric_limits<double>::max()));
      const T logMaxAmp = stk::math::log(maxAmp);
      const T logAmp = T(0.5) *
                       (stk::math::log(maxAbs) - stk::math::log(minAbsNonzero));
      const T amp =
        stk::math::max(T(1.0), stk::math::exp(stk::math::min(logAmp, logMaxAmp)));
      scale = maxAbs / amp;
    }
  }

  const T scaleSafe = stk::math::if_then_else(scale == T(0.0), T(1.0), scale);

  T B[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      B[i][j] = A[i][j] / scaleSafe;
    }
  }

  const T trA = B[0][0] + B[1][1] + B[2][2];
  const T shift = trA / 3.0;
  T C[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      C[i][j] = B[i][j];
    }
    C[i][i] -= shift;
  }

  const T p = C[0][0] * C[1][1] - C[0][1] * C[1][0] + C[1][1] * C[2][2] -
              C[1][2] * C[2][1] + C[0][0] * C[2][2] - C[0][2] * C[2][0];
  const T q =
    -(C[0][0] * C[1][1] * C[2][2] + C[0][1] * C[1][2] * C[2][0] +
      C[0][2] * C[1][0] * C[2][1] - C[0][0] * C[1][2] * C[2][1] -
      C[0][1] * C[1][0] * C[2][2] - C[0][2] * C[1][1] * C[2][0]);

  const T pTol =
    T(16.0) * T(std::numeric_limits<double>::epsilon()) *
    (stk::math::abs(C[0][0] * C[1][1]) + stk::math::abs(C[0][1] * C[1][0]) +
     stk::math::abs(C[1][1] * C[2][2]) + stk::math::abs(C[1][2] * C[2][1]) +
     stk::math::abs(C[0][0] * C[2][2]) + stk::math::abs(C[0][2] * C[2][0]));
  const T qAbs = stk::math::abs(q);
  const T qTol = T(16.0) * T(std::numeric_limits<double>::epsilon()) *
                 (stk::math::abs(C[0][0] * C[1][1] * C[2][2]) +
                  stk::math::abs(C[0][1] * C[1][2] * C[2][0]) +
                  stk::math::abs(C[0][2] * C[1][0] * C[2][1]) +
                  stk::math::abs(C[0][0] * C[1][2] * C[2][1]) +
                  stk::math::abs(C[0][1] * C[1][0] * C[2][2]) +
                  stk::math::abs(C[0][2] * C[1][1] * C[2][0]));
  const auto degenerate = (stk::math::abs(p) <= pTol) & (qAbs <= qTol);
  const T pSafe = stk::math::if_then_else(degenerate, T(-1.0), p);

  const T r = stk::math::sqrt(stk::math::max(-pSafe / 3.0, T(0.0)));
  const T qAbsSafe = stk::math::if_then_else(qAbs == T(0.0), T(1.0), qAbs);
  const T qSign = q / qAbsSafe;
  const T rSafe = stk::math::if_then_else(r > T(0.0), r, T(1.0));
  const T argLog = stk::math::log(qAbsSafe) - T(0.69314718055994530942) -
                   T(3.0) * stk::math::log(rSafe);
  const T argMag =
    stk::math::exp(stk::math::min(argLog, T(709.78271289338397)));
  const auto check_one =
    ((p > pTol) | ((p >= -pTol) & (qAbs > qTol))) |
    ((p < -pTol) & (qAbs > qTol) &
     (argMag > T(1.0) + T(16.0) * T(std::numeric_limits<double>::epsilon())));
  const bool exit_now = stk::simd::are_all(check_one & (!degenerate));
  if (exit_now) {
#if !defined(KOKKOS_ENABLE_GPU)
    KynemaUGFEnv::self().kynema_ugfOutput()
      << "Error, complex eigenvalues in EigenDecomposition::general_eigenvalues"
      << " p=" << p << " q=" << q << "([[" << A[0][0] << "," << A[0][1] << ","
      << A[0][2] << "],[" << A[1][0] << "," << A[1][1] << "," << A[1][2]
      << "],[" << A[2][0] << "," << A[2][1] << "," << A[2][2] << "]])"
      << std::endl;
    throw std::runtime_error(
      "ERROR, complex eigenvalues in EigenDecomposition::general_eigenvalues");
#else
    ThrowErrorMsgDevice(
      "ERROR, complex eigenvalues in EigenDecomposition::general_eigenvalues");
#endif
  }
  T arg = -qSign * argMag;
  arg = stk::math::min(stk::math::max(arg, T(-1.0)), T(1.0));

  const T phi = stk::math::acos(arg) / 3.0;

  const T t1 = 2.0 * r * stk::math::cos(phi);
  const T t2 = 2.0 * r * stk::math::cos(phi - 2.0 * pi / 3.0);
  const T t3 = 2.0 * r * stk::math::cos(phi - 4.0 * pi / 3.0);

  D[0][0] = stk::math::if_then_else(degenerate, shift, t1 + shift) * scale;
  D[1][1] = stk::math::if_then_else(degenerate, shift, t2 + shift) * scale;
  D[2][2] = stk::math::if_then_else(degenerate, shift, t3 + shift) * scale;
  D[0][1] = D[0][2] = D[1][0] = D[1][2] = D[2][0] = D[2][1] = T(0.0);
}

} // namespace EigenDecomposition

} // namespace kynema_ugf
} // namespace sierra

#endif

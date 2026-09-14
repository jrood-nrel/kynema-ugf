// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AMSUTILS_H
#define AMSUTILS_H

#include <SimdInterface.h>

namespace sierra {
namespace kynema_ugf {

namespace ams_utils {

template <class T, int dim = 3>
KOKKOS_FUNCTION T
get_M43_constant(T D[dim][dim], const double CMdeg)
{
  STK_NGP_ThrowRequireMsg(
    dim == 3, "Compute of M43 constant requires 3D problem");

  // Coefficients for the polynomial
  double c[15] = {0.971903113666644,  0.065591700544879,  0.071103489538998,
                  0.049918716158500,  -0.056904657182031, 0.097974249406576,
                  -0.015589487087603, 0.002003723064733,  0.002177318950949,
                  0.034227247973836,  0.001219656091495,  0.000417947294931,
                  0.000421085902741,  0.001223678414510,  0.003695127828465};

  const T eigScale = stk::math::max(
    stk::math::abs(D[0][0]),
    stk::math::max(stk::math::abs(D[1][1]), stk::math::abs(D[2][2])));
  const T eigFloor = stk::math::if_then_else(
    eigScale > T(0.0), T(1.0e-12) * eigScale, T(1.0e-16));
  const T d0 = stk::math::max(D[0][0], eigFloor);
  const T d1 = stk::math::max(D[1][1], eigFloor);
  const T d2 = stk::math::max(D[2][2], eigFloor);

  T smallestEV = stk::math::min(d0, stk::math::min(d1, d2));
  T largestEV = stk::math::max(d0, stk::math::max(d1, d2));
  T middleEV = stk::math::if_then_else(
    d0 == smallestEV, stk::math::min(d1, d2),
    stk::math::if_then_else(
      d1 == smallestEV, stk::math::min(d0, d2), stk::math::min(d0, d1)));

  // Scale the EVs
  middleEV = middleEV / smallestEV;
  largestEV = largestEV / smallestEV;

  const T middleSq = middleEV * middleEV;
  const T largestSq = largestEV * largestEV;
  const T r = stk::math::sqrt(middleSq + largestSq);
  const T sinTwoTheta = (T(2.0) * middleEV * largestEV) / (middleSq + largestSq);

  T x = stk::math::log(r);
  T y = stk::math::log(sinTwoTheta);

  T poly = c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y +
           c[5] * y * y + c[6] * x * x * x + c[7] * x * x * y +
           c[8] * x * y * y + c[9] * y * y * y + c[10] * x * x * x * x +
           c[11] * x * x * x * y + c[12] * x * x * y * y +
           c[13] * x * y * y * y + c[14] * y * y * y * y;

  return poly * CMdeg;
}

} // namespace ams_utils

} // namespace kynema_ugf
} // namespace sierra

#endif /* AMSUTILS_H */

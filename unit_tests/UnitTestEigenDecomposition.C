#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "EigenDecomposition.h"
#include "utils/AMSUtils.h"

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include "UnitTestUtils.h"

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include <master_element/MasterElement.h>

#include <fstream>

namespace {

std::random_device rd;
std::mt19937 rng(rd());
std::uniform_real_distribution<double> rand(0.0, 1.0);

double a11 = rand(rng);
double a22 = rand(rng);
double a33 = rand(rng);
double a12 = rand(rng);
double a13 = rand(rng);
double a23 = rand(rng);

double b11 = rand(rng);
double b22 = rand(rng);
double b33 = rand(rng);

static double A3d_rand[3][3] = {
  {a11, a12, a13},
  {a12, a22, a23},
  {a13, a23, a33},
};

static double A3d_fixed[3][3] = {
  {0.0562500000000000, 0.1875000000000000, 0.0125000000000000},
  {0.1875000000000000, 0.0093750000000000, -0.4218750000000000},
  {0.0125000000000000, -0.4218750000000000, 0.0031250000000000},
};

static double A2d_rand[2][2] = {
  {a11, a12},
  {a12, a22},
};

static double A2d_fixed[2][2] = {
  {0.0562500000000000, -0.0187500000000000},
  {-0.018750000000000, 0.0093750000000000},
};

DoubleType A3d_rand_simd[3][3];
DoubleType A3d_fixed_simd[3][3];
DoubleType A2d_rand_simd[2][2];
DoubleType A2d_fixed_simd[2][2];

double
max_abs_entry(const double (&A)[3][3])
{
  double maxAbs = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      maxAbs = std::max(maxAbs, std::abs(A[i][j]));
    }
  }
  return maxAbs;
}

void
expect_finite_real_roots(const double (&A)[3][3], const double (&D)[3][3])
{
  const double trA = A[0][0] + A[1][1] + A[2][2];
  const double detA =
    A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
    A[0][2] * A[1][0] * A[2][1] - A[0][0] * A[1][2] * A[2][1] -
    A[0][1] * A[1][0] * A[2][2] - A[0][2] * A[1][1] * A[2][0];
  const double coFacA = A[0][0] * A[1][1] - A[0][1] * A[1][0] +
                        A[1][1] * A[2][2] - A[1][2] * A[2][1] +
                        A[0][0] * A[2][2] - A[0][2] * A[2][0];

  const double matrixScale = max_abs_entry(A);
  const double polyTol = 1.0e-10 * matrixScale * matrixScale * matrixScale;
  const double traceTol = 1.0e-10 * matrixScale;

  double traceEval = 0.0;
  for (int i = 0; i < 3; ++i) {
    const double eig = D[i][i];
    EXPECT_TRUE(std::isfinite(eig));
    const double residual =
      eig * eig * eig - trA * eig * eig + coFacA * eig - detA;
    EXPECT_NEAR(residual, 0.0, polyTol);
    traceEval += eig;
  }

  EXPECT_NEAR(traceEval, trA, traceTol);
  EXPECT_DOUBLE_EQ(D[0][1], 0.0);
  EXPECT_DOUBLE_EQ(D[0][2], 0.0);
  EXPECT_DOUBLE_EQ(D[1][0], 0.0);
  EXPECT_DOUBLE_EQ(D[1][2], 0.0);
  EXPECT_DOUBLE_EQ(D[2][0], 0.0);
  EXPECT_DOUBLE_EQ(D[2][1], 0.0);
}

void
expect_device_real_roots(
  const double (&A)[3][3], const double (&expected)[3], const double tol)
{
  Kokkos::View<double[3][3], sierra::kynema_ugf::DeviceSpace> dA("dA");
  Kokkos::View<double[3][3], sierra::kynema_ugf::DeviceSpace> dD("dD");
  auto hA = Kokkos::create_mirror_view(dA);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      hA(i, j) = A[i][j];

  Kokkos::deep_copy(dA, hA);

  Kokkos::parallel_for(
    "test_general_eigenvalues_device",
    sierra::kynema_ugf::DeviceRangePolicy(0, 1), KOKKOS_LAMBDA(const int) {
      double localA[3][3], Q[3][3], D[3][3];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          localA[i][j] = dA(i, j);

      sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(localA, Q, D);

      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          dD(i, j) = D[i][j];
    });

  auto hD = Kokkos::create_mirror_view(dD);
  Kokkos::deep_copy(hD, dD);

  double eigenvalues[3] = {hD(0, 0), hD(1, 1), hD(2, 2)};
  std::sort(std::begin(eigenvalues), std::end(eigenvalues));

  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(std::isfinite(eigenvalues[i]));
    EXPECT_NEAR(eigenvalues[i], expected[i], tol);
  }
}

} // namespace

// This tests whether the correct eigenvalues are obtained
TEST(TestEigen, testeigendecomp3d)
{
  double Q_[3][3], D_[3][3];

  sierra::kynema_ugf::EigenDecomposition::sym_diagonalize(A3d_fixed, Q_, D_);

  // Perform tests -- lambda evaluated in Mathematica
  const double tol = 5.e-14;
  const double lambda_gold[3] = {
    0.056736539229635605, 0.46782517604126655, -0.45581171527090225};

  for (unsigned j = 0; j < 3; ++j) {
    EXPECT_NEAR(D_[j][j], lambda_gold[j], tol);
  }
}

TEST(TestEigen, testeigendecomp2d)
{
  double Q_[2][2], D_[2][2];

  sierra::kynema_ugf::EigenDecomposition::sym_diagonalize(A2d_fixed, Q_, D_);

  // Perform tests -- lambda evaluated in Mathematica
  const double tol = 5.e-14;
  const double lambda_gold[2] = {0.06282714486296648, 0.0027978551370335227};

  for (unsigned j = 0; j < 2; ++j) {
    EXPECT_NEAR(D_[j][j], lambda_gold[j], tol);
  }
}

// This tests that the eignevalue decomposition and reconstruction
// returns the original matrix
TEST(TestEigen, testeigendecompandreconstruct3d)
{
  double b_[3][3], Q_[3][3], D_[3][3];

  sierra::kynema_ugf::EigenDecomposition::sym_diagonalize(A3d_rand, Q_, D_);
  sierra::kynema_ugf::EigenDecomposition::reconstruct_matrix_from_decomposition(
    D_, Q_, b_);

  // Perform tests
  const double tol = 5.e-14;

  // Reconstructed matrix should be within tol of original
  for (unsigned j = 0; j < 3; ++j) {
    for (unsigned i = 0; i < 3; ++i) {
      EXPECT_NEAR(b_[i][j], A3d_rand[i][j], tol);
    }
  }
}

// This tests that the eignevalue decomposition and reconstruction
// returns the original matrix
TEST(TestEigen, testeigendecompandreconstruct2d)
{
  double b_[2][2], Q_[2][2], D_[2][2];

  sierra::kynema_ugf::EigenDecomposition::sym_diagonalize(A2d_rand, Q_, D_);
  sierra::kynema_ugf::EigenDecomposition::reconstruct_matrix_from_decomposition(
    D_, Q_, b_);

  // Perform tests
  const double tol = 5.e-14;

  // Reconstructed matrix should be within tol of original
  for (unsigned j = 0; j < 2; ++j) {
    for (unsigned i = 0; i < 2; ++i) {
      EXPECT_NEAR(b_[i][j], A2d_rand[i][j], tol);
    }
  }
}

// SIMD tests

// This tests whether the correct eigenvalues are obtained
TEST(TestEigen, testeigendecomp3d_simd)
{
  DoubleType Q_[3][3], D_[3][3];

  // Initialize matrices
  for (unsigned j = 0; j < stk::simd::ndoubles; ++j) {
    A3d_fixed_simd[0][0][j] = A3d_fixed[0][0] * (j + 1);
    A3d_fixed_simd[0][1][j] = A3d_fixed[0][1] * (j + 1);
    A3d_fixed_simd[0][2][j] = A3d_fixed[0][2] * (j + 1);
    A3d_fixed_simd[1][1][j] = A3d_fixed[1][1] * (j + 1);
    A3d_fixed_simd[1][2][j] = A3d_fixed[1][2] * (j + 1);
    A3d_fixed_simd[2][2][j] = A3d_fixed[2][2] * (j + 1);
  }
  A3d_fixed_simd[1][0] = A3d_fixed_simd[0][1];
  A3d_fixed_simd[2][0] = A3d_fixed_simd[0][2];
  A3d_fixed_simd[2][1] = A3d_fixed_simd[1][2];

  sierra::kynema_ugf::EigenDecomposition::sym_diagonalize(
    A3d_fixed_simd, Q_, D_);

  // Perform tests -- lambda evaluated in Mathematica
  const double tol = 5.e-14;
  const double lambda_gold[3] = {
    0.056736539229635605, 0.46782517604126655, -0.45581171527090225};

  for (unsigned j = 0; j < 3; ++j) {
    for (unsigned is = 0; is < stk::simd::ndoubles; is++) {
      EXPECT_NEAR(
        stk::simd::get_data(D_[j][j], is), (is + 1) * lambda_gold[j], tol);
    }
  }
}

TEST(TestEigen, testeigendecomp2d_simd)
{
  DoubleType Q_[2][2], D_[2][2];

  // Initialize matrices
  for (unsigned j = 0; j < stk::simd::ndoubles; ++j) {
    A2d_fixed_simd[0][0][j] = A2d_fixed[0][0] * (j + 1);
    A2d_fixed_simd[0][1][j] = A2d_fixed[0][1] * (j + 1);
    A2d_fixed_simd[1][1][j] = A2d_fixed[1][1] * (j + 1);
  }
  A2d_fixed_simd[1][0] = A2d_fixed_simd[0][1];

  sierra::kynema_ugf::EigenDecomposition::sym_diagonalize(
    A2d_fixed_simd, Q_, D_);

  // Perform tests -- lambda evaluated in Mathematica
  const double tol = 5.e-14;
  const double lambda_gold[2] = {0.06282714486296648, 0.0027978551370335227};

  for (unsigned j = 0; j < 2; ++j) {
    for (unsigned is = 0; is < stk::simd::ndoubles; is++) {
      EXPECT_NEAR(
        stk::simd::get_data(D_[j][j], is), (is + 1) * lambda_gold[j], tol);
    }
  }
}

// This tests that the eigenvalue decomposition and reconstruction
// returns the original matrix
TEST(TestEigen, testeigendecompandreconstruct3d_simd)
{
  DoubleType b_[3][3], Q_[3][3], D_[3][3];

  // Initialize matrices
  for (unsigned j = 0; j < stk::simd::ndoubles; ++j) {
    if (j % 2 == 0) {
      A3d_rand_simd[0][0][j] = a11 * (j + 1);
      A3d_rand_simd[0][1][j] = a12 * (j + 1);
      A3d_rand_simd[0][2][j] = a13 * (j + 1);
      A3d_rand_simd[1][1][j] = a22 * (j + 1);
      A3d_rand_simd[1][2][j] = a23 * (j + 1);
      A3d_rand_simd[2][2][j] = a33 * (j + 1);
    }
    // Test if some of the simd entries are already diagonal
    else {
      A3d_rand_simd[0][0][j] = b11 * (j + 1);
      A3d_rand_simd[0][1][j] = 0.0;
      A3d_rand_simd[0][2][j] = 0.0;
      A3d_rand_simd[1][1][j] = b22 * (j + 1);
      A3d_rand_simd[1][2][j] = 0.0;
      A3d_rand_simd[2][2][j] = b33 * (j + 1);
    }
  }
  A3d_rand_simd[1][0] = A3d_rand_simd[0][1];
  A3d_rand_simd[2][0] = A3d_rand_simd[0][2];
  A3d_rand_simd[2][1] = A3d_rand_simd[1][2];

  sierra::kynema_ugf::EigenDecomposition::sym_diagonalize(
    A3d_rand_simd, Q_, D_);
  sierra::kynema_ugf::EigenDecomposition::reconstruct_matrix_from_decomposition(
    D_, Q_, b_);

  // Perform tests
  const double tol = 5.e-14;

  // Reconstructed matrix should be within tol of original
  for (unsigned j = 0; j < 3; ++j) {
    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned is = 0; is < stk::simd::ndoubles; is++) {
        EXPECT_NEAR(
          stk::simd::get_data(b_[i][j], is),
          stk::simd::get_data(A3d_rand_simd[i][j], is), tol);
      }
    }
  }
}

TEST(TestEigen, testeigendecompandreconstruct2d_simd)
{
  DoubleType b_[2][2], Q_[2][2], D_[2][2];

  // Initialize matrices
  for (unsigned j = 0; j < stk::simd::ndoubles; ++j) {
    if (j % 2 == 0) {
      A2d_rand_simd[0][0][j] = a11 * (j + 1);
      A2d_rand_simd[0][1][j] = a12 * (j + 1);
      A2d_rand_simd[1][1][j] = a22 * (j + 1);
    }
    // Test if some of the simd entries are already diagonal
    else {
      A2d_rand_simd[0][0][j] = b11 * (j + 1);
      A2d_rand_simd[0][1][j] = 0.0;
      A2d_rand_simd[1][1][j] = b22 * (j + 1);
    }
  }

  A2d_rand_simd[1][0] = A2d_rand_simd[0][1];

  sierra::kynema_ugf::EigenDecomposition::sym_diagonalize(
    A2d_rand_simd, Q_, D_);
  sierra::kynema_ugf::EigenDecomposition::reconstruct_matrix_from_decomposition(
    D_, Q_, b_);

  // Perform tests
  const double tol = 5.e-14;

  // Reconstructed matrix should be within tol of original
  for (unsigned j = 0; j < 2; ++j) {
    for (unsigned i = 0; i < 2; ++i) {
      for (unsigned is = 0; is < stk::simd::ndoubles; is++) {
        EXPECT_NEAR(
          stk::simd::get_data(b_[i][j], is),
          stk::simd::get_data(A2d_rand_simd[i][j], is), tol);
      }
    }
  }
}

TEST(TestEigen, testgeneraleigenvalues_robust_cases)
{
  double Q_[3][3], D_[3][3];

  const double zero[3][3] = {
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0},
  };

  const double tripleRoot[3][3] = {
    {7.0, 0.0, 0.0},
    {0.0, 7.0, 0.0},
    {0.0, 0.0, 7.0},
  };

  const double doubleRoot[3][3] = {
    {2.0, 0.0, 0.0},
    {0.0, 2.0, 0.0},
    {0.0, 0.0, 5.0},
  };
  const double nearTripleRoot[3][3] = {
    {1.0 - 5.0e-7, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0 + 5.0e-7},
  };

  const double nearZero[3][3] = {
    {2.0e-20, -1.0e-20, 5.0e-21},
    {-1.0e-20, 2.0e-20, 2.5e-21},
    {5.0e-21, 2.5e-21, 1.5e-20},
  };

  const double nearLarge[3][3] = {
    {2.0e20, -1.0e20, 5.0e19},
    {-1.0e20, 2.0e20, 2.5e19},
    {5.0e19, 2.5e19, 1.5e20},
  };
  const double nonsymmetricReal[3][3] = {
    {3.0, 1.0, 0.0},
    {0.0, 2.0, 1.0},
    {0.0, 0.0, 1.0},
  };
  const double subnormalReal[3][3] = {
    {1.0e-160, 1.0, 0.0},
    {0.0, -1.0e-160, 0.0},
    {0.0, 0.0, 0.0},
  };
  const double unbalancedReal[3][3] = {
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0e-200},
    {0.0, 1.0e-200, 0.0},
  };
  const double dynamicRangeReal[3][3] = {
    {1.0, 1.0e-320, 0.0},
    {0.0, 2.0, 0.0},
    {0.0, 0.0, 4.0},
  };
  const double cappedAmplificationGap[3][3] = {
    {0.0, 1.0e100, 0.0},
    {0.0, 0.0, 1.0e-300},
    {0.0, 1.0e-300, 0.0},
  };

  double A[3][3];
  const double(*cases[])[3] = {
    zero,      tripleRoot,       doubleRoot,    nearTripleRoot, nearZero,
    nearLarge, nonsymmetricReal, subnormalReal, unbalancedReal};
  for (const auto& testCase : cases) {
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        A[i][j] = testCase[i][j];

    sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q_, D_);
    expect_finite_real_roots(A, D_);
  }

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      A[i][j] = nearTripleRoot[i][j];

  sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q_, D_);

  {
    double eigenvalues[3] = {D_[0][0], D_[1][1], D_[2][2]};
    std::sort(std::begin(eigenvalues), std::end(eigenvalues));
    EXPECT_NEAR(eigenvalues[0], 1.0 - 5.0e-7, 1.0e-12);
    EXPECT_NEAR(eigenvalues[1], 1.0, 1.0e-12);
    EXPECT_NEAR(eigenvalues[2], 1.0 + 5.0e-7, 1.0e-12);
  }

  const double complexPair[3][3] = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, -1.0e-5},
    {0.0, 1.0, 0.0},
  };
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      A[i][j] = complexPair[i][j];

  EXPECT_THROW(
    sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q_, D_),
    std::runtime_error);

  const double shiftedComplexPair[3][3] = {
    {1.0, 0.0, -1.0e-15},
    {1.0, 1.0, 0.0},
    {0.0, 1.0, 1.0},
  };
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      A[i][j] = shiftedComplexPair[i][j];

  EXPECT_THROW(
    sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q_, D_),
    std::runtime_error);

  const double tinyComplexPair[3][3] = {
    {0.0, 0.0, -1.0e-200},
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
  };
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      A[i][j] = tinyComplexPair[i][j];

  EXPECT_THROW(
    sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q_, D_),
    std::runtime_error);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      A[i][j] = unbalancedReal[i][j];

  sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q_, D_);

  {
    double eigenvalues[3] = {D_[0][0], D_[1][1], D_[2][2]};
    std::sort(std::begin(eigenvalues), std::end(eigenvalues));
    EXPECT_NEAR(eigenvalues[0], -1.0e-200, 1.0e-212);
    EXPECT_NEAR(eigenvalues[1], 0.0, 1.0e-212);
    EXPECT_NEAR(eigenvalues[2], 1.0e-200, 1.0e-212);
  }

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      A[i][j] = dynamicRangeReal[i][j];

  sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q_, D_);

  {
    double eigenvalues[3] = {D_[0][0], D_[1][1], D_[2][2]};
    std::sort(std::begin(eigenvalues), std::end(eigenvalues));
    EXPECT_NEAR(eigenvalues[0], 1.0, 1.0e-12);
    EXPECT_NEAR(eigenvalues[1], 2.0, 1.0e-12);
    EXPECT_NEAR(eigenvalues[2], 4.0, 1.0e-12);
  }

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      A[i][j] = cappedAmplificationGap[i][j];

  sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q_, D_);

  {
    double eigenvalues[3] = {D_[0][0], D_[1][1], D_[2][2]};
    std::sort(std::begin(eigenvalues), std::end(eigenvalues));
    EXPECT_NEAR(eigenvalues[0], -1.0e-300, 1.0e-312);
    EXPECT_NEAR(eigenvalues[1], 0.0, 1.0e-312);
    EXPECT_NEAR(eigenvalues[2], 1.0e-300, 1.0e-312);
  }

  const double cappedAmplificationGapExpected[3] = {-1.0e-300, 0.0, 1.0e-300};
  expect_device_real_roots(
    cappedAmplificationGap, cappedAmplificationGapExpected, 1.0e-312);
}

TEST(TestAMSUtils, testgetm43constant_finite_for_subnormal_spectrum)
{
  double D[3][3] = {
    {1.0e-320, 0.0, 0.0},
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0},
  };

  const double cm43 =
    sierra::kynema_ugf::ams_utils::get_M43_constant<double>(D, 1.0);
  EXPECT_TRUE(std::isfinite(cm43));
}

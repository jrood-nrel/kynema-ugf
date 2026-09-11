#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "EigenDecomposition.h"

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

TEST(TestEigen, testgeneraleigenvalueszeromatrix)
{
  double A[3][3] = {
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0},
  };
  double Q[3][3], D[3][3];

  sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q, D);

  for (unsigned j = 0; j < 3; ++j) {
    EXPECT_DOUBLE_EQ(D[j][j], 0.0);
    EXPECT_TRUE(std::isfinite(D[j][j]));
  }
}

TEST(TestEigen, testgeneraleigenvaluestriplenonzeroeigenvalue)
{
  constexpr double c = 1.1;
  double A[3][3] = {
    {c, 0.0, 0.0},
    {0.0, c, 0.0},
    {0.0, 0.0, c},
  };
  double Q[3][3], D[3][3];

  sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q, D);

  for (unsigned j = 0; j < 3; ++j) {
    EXPECT_NEAR(D[j][j], c, 1.0e-14);
    EXPECT_TRUE(std::isfinite(D[j][j]));
  }
}

TEST(TestEigen, testgeneraleigenvaluestranslationinvariantthreshold)
{
  double A[3][3] = {
    {1.0e8 - 1.0, 0.0, 0.0},
    {0.0, 1.0e8, 0.0},
    {0.0, 0.0, 1.0e8 + 1.0},
  };
  double Q[3][3], D[3][3];

  sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q, D);

  std::array<double, 3> eigenvalues = {D[0][0], D[1][1], D[2][2]};
  std::sort(eigenvalues.begin(), eigenvalues.end());
  constexpr std::array<double, 3> expected = {1.0e8 - 1.0, 1.0e8, 1.0e8 + 1.0};

  for (unsigned j = 0; j < 3; ++j) {
    EXPECT_NEAR(eigenvalues[j], expected[j], 1.0e-8);
  }
}

TEST(TestEigen, testgeneraleigenvaluesdistinctrealroots)
{
  double A[3][3] = {
    {3.0, 0.0, 0.0},
    {0.0, -1.0, 0.0},
    {0.0, 0.0, 2.0},
  };
  double Q[3][3], D[3][3];

  sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q, D);

  std::array<double, 3> eigenvalues = {D[0][0], D[1][1], D[2][2]};
  std::sort(eigenvalues.begin(), eigenvalues.end());
  constexpr std::array<double, 3> expected = {-1.0, 2.0, 3.0};

  for (unsigned j = 0; j < 3; ++j) {
    EXPECT_NEAR(eigenvalues[j], expected[j], 1.0e-13);
    EXPECT_TRUE(std::isfinite(eigenvalues[j]));
  }
}

TEST(TestEigen, testgeneraleigenvaluesmixedlanes_simd)
{
  DoubleType A[3][3], Q[3][3], D[3][3];

  for (unsigned lane = 0; lane < stk::simd::ndoubles; ++lane) {
    if (lane % 2 == 0) {
      A[0][0][lane] = 0.0;
      A[1][1][lane] = 0.0;
      A[2][2][lane] = 0.0;
    }
    else {
      A[0][0][lane] = -1.0e-151;
      A[1][1][lane] = 0.0;
      A[2][2][lane] = 1.0e-151;
    }
    A[0][1][lane] = 0.0;
    A[0][2][lane] = 0.0;
    A[1][0][lane] = 0.0;
    A[1][2][lane] = 0.0;
    A[2][0][lane] = 0.0;
    A[2][1][lane] = 0.0;
  }

  sierra::kynema_ugf::EigenDecomposition::general_eigenvalues(A, Q, D);

  for (unsigned lane = 0; lane < stk::simd::ndoubles; ++lane) {
    std::array<double, 3> eigenvalues = {
      stk::simd::get_data(D[0][0], lane),
      stk::simd::get_data(D[1][1], lane),
      stk::simd::get_data(D[2][2], lane)};
    std::sort(eigenvalues.begin(), eigenvalues.end());

    if (lane % 2 == 0) {
      EXPECT_DOUBLE_EQ(eigenvalues[0], 0.0);
      EXPECT_DOUBLE_EQ(eigenvalues[1], 0.0);
      EXPECT_DOUBLE_EQ(eigenvalues[2], 0.0);
    }
    else {
      EXPECT_NEAR(eigenvalues[0], -1.0e-151, 1.0e-163);
      EXPECT_NEAR(eigenvalues[1], 0.0, 1.0e-300);
      EXPECT_NEAR(eigenvalues[2], 1.0e-151, 1.0e-163);
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

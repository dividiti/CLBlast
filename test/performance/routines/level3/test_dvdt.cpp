
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// =================================================================================================

#include "test/performance/client.hpp"
#include "test/routines/level3/test_dvdt.hpp"

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args, clblast::Precision::kSingle)) {
    case clblast::Precision::kHalf:
      clblast::RunClient<clblast::TestDvdt<clblast::half>, clblast::half, clblast::half>(argc, argv); break;
    case clblast::Precision::kSingle:
      clblast::RunClient<clblast::TestDvdt<float>, float, float>(argc, argv); break;
    case clblast::Precision::kDouble:
      clblast::RunClient<clblast::TestDvdt<double>, double, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle:
      clblast::RunClient<clblast::TestDvdt<clblast::float2>, clblast::float2, clblast::float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble:
      clblast::RunClient<clblast::TestDvdt<clblast::double2>, clblast::double2, clblast::double2>(argc, argv); break;
  }
  return 0;
}

// =================================================================================================


// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the interface to the CLTune auto-tuner. This is only used for the optional
// and stand-alone tuner binaries and not part of the core of CLBlast.
//
// =================================================================================================

#ifndef CLBLAST_TUNING_H_
#define CLBLAST_TUNING_H_

#include <vector>
#include <string>
#include <random>

#include <cltune.h>

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Function to get command-line argument, set-up the input buffers, configure the tuner, and collect
// the results. Used for all types of kernel families. Note that this is a header-only function so
// that it is automatically compiled for the various kernels (given as the 'C' template argument).
template <typename C, typename T>
void Tuner(int argc, char* argv[]) {
  constexpr auto kSeed = 42; // fixed seed for reproducibility

  // Sets the parameters and platform/device for which to tune (command-line options)
  auto command_line_args = RetrieveCommandLineArguments(argc, argv);
  auto help = std::string{"* Options given/available:\n"};
  auto args = Arguments<T>{};
  args.platform_id = GetArgument(command_line_args, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  args.device_id   = GetArgument(command_line_args, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  args.precision   = GetArgument(command_line_args, help, kArgPrecision, Precision::kSingle);
  for (auto &o: C::GetOptions()) {
    if (o == kArgM)        { args.m        = GetArgument(command_line_args, help, kArgM, C::DefaultM()); }
    if (o == kArgN)        { args.n        = GetArgument(command_line_args, help, kArgN, C::DefaultN()); }
    if (o == kArgK)        { args.k        = GetArgument(command_line_args, help, kArgK, C::DefaultK()); }
    if (o == kArgAlpha)    { args.alpha    = GetArgument(command_line_args, help, kArgAlpha, GetScalar<T>()); }
    if (o == kArgBeta)     { args.beta     = GetArgument(command_line_args, help, kArgBeta, GetScalar<T>()); }
    if (o == kArgFraction) { args.fraction = GetArgument(command_line_args, help, kArgFraction, C::DefaultFraction()); }
    if (o == kArgBatchCount) { args.batch_count = GetArgument(command_line_args, help, kArgBatchCount, C::DefaultBatchCount()); }
    if (o == tStrategy)   {args.tStrategy   = GetArgument(command_line_args, help, tStrategy, DEFAULT_STRATEGY);  }
    if (o == psoSwarmSize)   {args.psoSwarmSize   = GetArgument(command_line_args, help, psoSwarmSize, DEFAULT_PSO_SWARM);  }
    if (o == psoInfG)   {args.psoInfG   = GetArgument(command_line_args, help, psoInfG, DEFAULT_PSO_G);  }
    if (o == psoInfL)   {args.psoInfL   = GetArgument(command_line_args, help, psoInfL, DEFAULT_PSO_L);  }
    if (o == psoInfR)   {args.psoInfR   = GetArgument(command_line_args, help, psoInfR, DEFAULT_PSO_R);  }
  }
  const auto num_runs = GetArgument(command_line_args, help, kArgNumRuns, C::DefaultNumRuns());

  fprintf(stdout, "%s\n", help.c_str());

  // Tests validity of the given arguments
  C::TestValidArguments(args);

  // Tests for validity of the precision and retrieves properties
  auto isAMD = false;
  auto isARM = false;
  auto isGPU = false;
  {
    const auto platform = Platform(args.platform_id);
    const auto device = Device(platform, args.device_id);
    if (!PrecisionSupported<T>(device)) {
      printf("* Unsupported precision, skipping this tuning run\n\n");
      return;
    }
    isAMD = device.IsAMD();
    isARM = device.IsARM();
    isGPU = device.IsGPU();
  }

  // Creates input buffers with random data
  auto x_vec = std::vector<T>(C::GetSizeX(args));
  auto y_vec = std::vector<T>(C::GetSizeY(args));
  auto a_mat = std::vector<T>(C::GetSizeA(args));
  auto b_mat = std::vector<T>(C::GetSizeB(args));
  auto c_mat = std::vector<T>(C::GetSizeC(args));
  auto temp = std::vector<T>(C::GetSizeTemp(args));
  std::mt19937 mt(kSeed);
  std::uniform_real_distribution<double> dist(kTestDataLowerLimit, kTestDataUpperLimit);
  PopulateVector(x_vec, mt, dist);
  PopulateVector(y_vec, mt, dist);
  PopulateVector(a_mat, mt, dist);
  PopulateVector(b_mat, mt, dist);
  PopulateVector(c_mat, mt, dist);
  PopulateVector(temp, mt, dist);

  // Initializes the tuner for the chosen device
  cltune::Tuner tuner(args.platform_id, args.device_id);

  // Use full-search to explore all parameter combinations or random-search to search only a part of
  // the parameter values. The fraction is set as a command-line argument.
  #ifdef XGEMM_EXEC
  
  if(tStrategyFlag)
  {
   auto localtStrategy = args.tStrategy;  

    if (args.fraction == 1.0 || args.fraction == 0.0) 
    { 
     localtStrategy = FULL_SEARCH_STRATEGY; 
    }
    switch (localtStrategy)
    {
      case FULL_SEARCH_STRATEGY: 
        tuner.UseFullSearch();
        break;

      case RANDOM_SEARCH_STRATEGY: 
          tuner.UseRandomSearch(1.0/args.fraction);
        break;
      case PSO_STRATEGY: 
          tuner.UsePSO(1.0/args.fraction, args.psoSwarmSize, args.psoInfG, args.psoInfL, args.psoInfR);
        break;
      case DVDT_STRATEGY:
      default: 
        tuner.UseFullSearch();
    }
  }

  #else

  if (args.fraction == 1.0 || args.fraction == 0.0) 
  {
    tuner.UseFullSearch();
  }
  else 
  {
    tuner.UseRandomSearch(1.0/args.fraction);
  }

  #endif
  // Set extra settings for specific defines. This mimics src/routine.cc.
  auto defines = std::string{""};
  if (isAMD && isGPU) {
    defines += "#define USE_CL_MAD 1\n";
    defines += "#define USE_STAGGERED_INDICES 1\n";
  }
  if (isARM && isGPU) {
    defines += "#define GLOBAL_MEM_FENCE 1\n";
  }

  // Loads the kernel sources and defines the kernel to tune
  auto sources = defines + C::GetSources();
  auto id = tuner.AddKernelFromString(sources, C::KernelName(), C::GlobalSize(args), C::LocalSize());
  tuner.SetReferenceFromString(sources, C::KernelName(), C::GlobalSizeRef(args), C::LocalSizeRef());

  // Sets the tunable parameters and their possible values
  C::SetParameters(tuner, id);
  C::SetConstraints(tuner, id);
  C::SetLocalMemorySize(tuner, id, args);

  // Tests for a specific precision
  tuner.AddParameter(id, "PRECISION", {static_cast<size_t>(args.precision)});
  tuner.AddParameterReference("PRECISION", static_cast<size_t>(args.precision));

  // Modifies the thread-sizes (both global and local) based on the parameters
  for (auto &parameters: C::MulLocal()) { tuner.MulLocalSize(id, parameters); }
  for (auto &parameters: C::DivLocal()) { tuner.DivLocalSize(id, parameters); }
  for (auto &parameters: C::MulGlobal()) { tuner.MulGlobalSize(id, parameters); }
  for (auto &parameters: C::DivGlobal()) { tuner.DivGlobalSize(id, parameters); }

  // Sets the function's arguments
  C::SetArguments(tuner, args, x_vec, y_vec, a_mat, b_mat, c_mat, temp);

  // Starts the tuning process
  tuner.SetNumRuns(num_runs);
  tuner.Tune();

  // Prints the results to screen
  auto time_ms = tuner.PrintToScreen();
  tuner.PrintFormatted();

  // Also prints the performance of the best-case in terms of GB/s or GFLOPS
  if (time_ms != 0.0) {
    printf("[ -------> ] %.2lf ms", time_ms);
    printf(" or %.1lf %s\n", C::GetMetric(args)/(time_ms*1.0e6), C::PerformanceUnit().c_str());
  }

  // Outputs the results as JSON to disk, including some meta-data
  auto precision_string = std::to_string(static_cast<size_t>(args.precision));
  auto metadata = std::vector<std::pair<std::string,std::string>>{
    {"kernel_family", C::KernelFamily()},
    {"precision", precision_string}
  };
  for (auto &o: C::GetOptions()) {
    if (o == kArgM)     { metadata.push_back({"arg_m", std::to_string(args.m)}); }
    if (o == kArgN)     { metadata.push_back({"arg_n", std::to_string(args.n)}); }
    if (o == kArgK)     { metadata.push_back({"arg_k", std::to_string(args.k)}); }
    if (o == kArgAlpha) { metadata.push_back({"arg_alpha", ToString(args.alpha)}); }
    if (o == kArgBeta)  { metadata.push_back({"arg_beta", ToString(args.beta)}); }
    if (o == kArgBatchCount) { metadata.push_back({"arg_batch_count", ToString(args.batch_count)}); }
  }
  tuner.PrintJSON("clblast_"+C::KernelFamily()+"_"+precision_string+".json", metadata);
 

}

// =================================================================================================
} // namespace clblast

// CLBLAST_TUNING_H_
#endif

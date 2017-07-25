
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the tests for the OverrideParameters function
//
// =================================================================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <chrono>
#include <limits>
#include "utilities/utilities.hpp"
#include "test/routines/level3/test_dvdt.hpp"

namespace clblast {
constexpr auto kArgKwg = "KWG";

constexpr auto kArgKwi = "KWI";

constexpr auto kArgMdima = "MDIMA";

constexpr auto kArgMdimc = "MDIMC";

constexpr auto kArgMwg = "MWG";

constexpr auto kArgNdimb = "NDIMB";

constexpr auto kArgNdimc = "NDIMC";

constexpr auto kArgNwg = "NWG";

constexpr auto kArgSa = "SA";

constexpr auto kArgSb = "SB";

constexpr auto kArgStrm = "STRM";

constexpr auto kArgStrn = "STRN";

constexpr auto kArgVwm = "VWM";

constexpr auto kArgVwn = "VWN";

constexpr auto KERNEL_NAME = "kernel_name";

// =================================================================================================

template <typename T>
size_t RunOverrideTestsDvdt(int argc, char *argv[], const bool silent, const std::string &routine_name) {
  auto arguments = RetrieveCommandLineArguments(argc, argv);

  auto errors = size_t{0};
  auto passed = size_t{0};
  auto example_routine = TestDvdt<T>();
  constexpr auto kSeed = 42; // fixed seed for reproducibility

  // Determines the test settings
  const auto kernel_name = std::string{"Xgemm"};
  const auto precision = PrecisionValue<T>();
  

  // Retrieves the arguments
  auto help = std::string{"Options given/available:\n"};
  const auto platform_id = GetArgument(arguments, help, kArgPlatform, ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  const auto device_id = GetArgument(arguments, help, kArgDevice, ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  auto args = Arguments<T>{};
  args.m = GetArgument(arguments, help, kArgM, size_t{256});
  args.n = GetArgument(arguments, help, kArgN, size_t{256});
  args.k = GetArgument(arguments, help, kArgK, size_t{256});
  args.a_ld = GetArgument(arguments, help, kArgALeadDim, args.k);
  args.b_ld = GetArgument(arguments, help, kArgBLeadDim, args.n);
  args.c_ld = GetArgument(arguments, help, kArgCLeadDim, args.n);
  args.a_offset = GetArgument(arguments, help, kArgAOffset, size_t{0});
  args.b_offset = GetArgument(arguments, help, kArgBOffset, size_t{0});
  args.c_offset = GetArgument(arguments, help, kArgCOffset, size_t{0});
  args.layout = GetArgument(arguments, help, kArgLayout, Layout::kRowMajor);
  args.a_transpose = GetArgument(arguments, help, kArgATransp, Transpose::kNo);
  args.b_transpose = GetArgument(arguments, help, kArgBTransp, Transpose::kNo);
  args.alpha = GetArgument(arguments, help, kArgAlpha, GetScalar<T>());
  args.beta  = GetArgument(arguments, help, kArgBeta, GetScalar<T>());
  

  auto kwg = GetArgument(arguments, help, kArgKwg, size_t{32});
  auto kwi = GetArgument(arguments, help, kArgKwi, size_t{2});
  auto mdima = GetArgument(arguments, help, kArgMdima, size_t{8});
  auto mdimc = GetArgument(arguments, help, kArgMdimc, size_t{8});
  auto mwg = GetArgument(arguments, help, kArgMwg, size_t{64});
  auto ndimb = GetArgument(arguments, help, kArgNdimb, size_t{16});
  auto ndimc = GetArgument(arguments, help, kArgNdimc, size_t{16});
  auto nwg = GetArgument(arguments, help, kArgNwg, size_t{64});
  auto sa = GetArgument(arguments, help, kArgSa, size_t{1});
  auto sb = GetArgument(arguments, help, kArgSb, size_t{1});
  auto strm = GetArgument(arguments, help, kArgStrm, size_t{0});
  auto strn = GetArgument(arguments, help, kArgStrn, size_t{0});
  auto vwm = GetArgument(arguments, help, kArgVwm, size_t{4});
  auto vwn = GetArgument(arguments, help, kArgVwn, size_t{4});
  auto num_runs = GetArgument(arguments, help, kArgNumRuns, size_t{5});
  std::string k_name = GetStringArgument(arguments,help, KERNEL_NAME, "Xgemm");
  
  
  const auto cmdline_settings = std::unordered_map<std::string,size_t>{
   {kArgKwg, kwg}, {kArgKwi, kwi}, {kArgMdima, mdima}, {kArgMdimc, mdimc},
   {kArgMwg, mwg}, {kArgNdimb, ndimb}, {kArgNdimc, ndimc}, {kArgNwg, nwg},
   {kArgSa, sa}, {kArgSb, sb}, {kArgStrm, strm}, {kArgStrn, strn}, {kArgVwm, vwm}, 
   {kArgVwn,vwn}  
 };
  // Prints the help message (command-line arguments)
  if (!silent) { fprintf(stdout, "\n* %s\n", help.c_str()); }
  
  
  // Initializes OpenCL
  const auto platform = Platform(platform_id);
  const auto device = Device(platform, device_id);
  const auto context = Context(device);
  auto queue = Queue(context, device);

  // Populate host matrices with some example data
  auto host_a = std::vector<T>(args.m * args.k);
  auto host_b = std::vector<T>(args.n * args.k);
  auto host_c = std::vector<T>(args.m * args.n);
  std::mt19937 mt(kSeed);
  std::uniform_real_distribution<double> dist(kTestDataLowerLimit, kTestDataUpperLimit);
  PopulateVector(host_a, mt, dist);
  PopulateVector(host_b, mt, dist);
  PopulateVector(host_c, mt, dist);

  // Copy the matrices to the device
  auto device_a = Buffer<T>(context, host_a.size());
  auto device_b = Buffer<T>(context, host_b.size());
  auto device_c = Buffer<T>(context, host_c.size());
  device_a.Write(queue, host_a.size(), host_a);
  device_b.Write(queue, host_b.size(), host_b);
  device_c.Write(queue, host_c.size(), host_c);
  auto dummy = Buffer<T>(context, 1);
  auto buffers = Buffers<T>{dummy, dummy, device_a, device_b, device_c, dummy, dummy};

  // Loops over the valid combinations: run before and run afterwards
  fprintf(stdout, "* Testing OverrideParameters for '%s'\n", routine_name.c_str());
  // for (const auto &override_setting : cmdline_settings) {
  const auto status_before = example_routine.RunRoutine(args, buffers, queue);
  if (status_before != StatusCode::kSuccess) { errors++; }

    // Overrides the parameters
  const auto status = OverrideParameters(device(), kernel_name, precision, cmdline_settings);
  if (status != StatusCode::kSuccess) { errors++; fprintf(stderr, "ERROR : %d - %zu\n",status,errors ); } // error shouldn't occur


  const auto status_after = example_routine.RunRoutine(args, buffers, queue);
  if (status_after != StatusCode::kSuccess) { errors++; fprintf(stderr, "%zu\n",errors ); }
  passed++;

  if(!errors)
  {
    auto elapsed_time = std::numeric_limits<float>::max();
    
    for (auto i = 0; i < num_runs; i++) {
      const auto start_time = std::chrono::steady_clock::now();

      const auto status_after = example_routine.RunRoutine(args, buffers, queue);
       queue.Finish();
      auto cpu_timer = std::chrono::steady_clock::now() - start_time;

      if (status_after != StatusCode::kSuccess) { errors++;}
     const auto cpu_timing = std::chrono::duration<float,std::milli>(cpu_timer).count();
        elapsed_time = std::min(elapsed_time, cpu_timing);
      
    }
    fprintf(stderr, "Elapsed_time : %.1lf(ms)\n", elapsed_time );
    long long int flops = (long long int) args.m * (long long int) args.n * (long long int) args.k * 2;
    auto gflops = (elapsed_time != 0.0) ? (flops*1e-6)/elapsed_time : 0;
    fprintf(stderr, "GFLOPS : %9.1lf\n",gflops);

    //Print Json
    std::stringstream tmp_filename;
    tmp_filename << "clblast_xgemm_override.json";
    std::string filename = tmp_filename.str();
    FILE * f = fopen(filename.c_str(),"w");
    fprintf(f, "{\n" );
    fprintf(f, "  \"m\" : %u,\n", args.m);
    fprintf(f, "  \"n\" : %u,\n", args.n);
    fprintf(f, "  \"k\" : %u,\n", args.k);
    fprintf(f, "  \"GFLOPS\" : %lf,\n", gflops);
    fprintf(f, "  \"Time(ms)\" : %.4lf,\n", elapsed_time);
    fprintf(f, "  \"kernel_family\" : \"%s\",\n", k_name.c_str());
    fprintf(f, "  \"device\": \"%s\",\n", device.Name().c_str());
    fprintf(f, "  \"device_vendor\": \"%s\",\n", device.Vendor().c_str());
    fprintf(f, "  \"device_type\": \"%s\",\n", device.Type().c_str());
    fprintf(f, "  \"device_core_clock\": \"%zu\",\n", device.CoreClock());
    fprintf(f, "  \"device_compute_units\": \"%zu\"\n", device.ComputeUnits());
    fprintf(f, "}" );

    fclose(f);

  }

  
  return errors;
}

// =================================================================================================
} // namespace clblast

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  auto errors = size_t{0};
  errors += clblast::RunOverrideTestsDvdt<float>(argc, argv, false, "SGEMM");
  if (errors > 0) { return 1; } else { return 0; }
}

// =================================================================================================

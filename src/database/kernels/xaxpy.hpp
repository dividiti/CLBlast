
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xaxpy' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XaxpyHalf = {
  "Xaxpy", Precision::kHalf, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "Ellesmere",                                       { {"VW",4}, {"WGS",128}, {"WPT",4} } },
        { "default",                                         { {"VW",4}, {"WGS",128}, {"WPT",4} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        // { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        // { "Intel(R) HD Graphics Skylake ULT GT2",            { {"VW",8}, {"WGS",64}, {"WPT",1} } },
        { "default",                                         { {"VW",8}, {"WGS",64}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW",8}, {"WGS",256}, {"WPT",4} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XaxpySingle = {
  "Xaxpy", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "AMD Radeon R9 M370X Compute Engine",              { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        // { "ATI Radeon HD 6750M",                             { {"VW",1}, {"WGS",256}, {"WPT",2} } },
        // { "Ellesmere",                                       { {"VW",1}, {"WGS",64}, {"WPT",4} } },
        // { "Fiji",                                            { {"VW",4}, {"WGS",64}, {"WPT",1} } },
        // { "Hawaii",                                          { {"VW",2}, {"WGS",64}, {"WPT",2} } },
        // { "Oland",                                           { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        // { "Pitcairn",                                        { {"VW",2}, {"WGS",128}, {"WPT",1} } },
        // { "Tahiti",                                          { {"VW",2}, {"WGS",64}, {"WPT",1} } },
        // { "Tonga",                                           { {"VW",1}, {"WGS",256}, {"WPT",8} } },
        // { "Turks",                                           { {"VW",2}, {"WGS",256}, {"WPT",1} } },
        { "default",                                         { {"VW",2}, {"WGS",256}, {"WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"VW",4}, {"WGS",256}, {"WPT",1} } },
        { "default",                                         { {"VW",4}, {"WGS",256}, {"WPT",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        // { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"VW",8}, {"WGS",512}, {"WPT",1} } },
        // { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW",1}, {"WGS",512}, {"WPT",1} } },
        // { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW",4}, {"WGS",256}, {"WPT",1} } },
        // { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"VW",2}, {"WGS",1024}, {"WPT",1} } },
        // { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "default",                                         { {"VW",8}, {"WGS",512}, {"WPT",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        // { "Intel(R) HD Graphics 530",                        { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        // { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        // { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        // { "Intel(R) HD Graphics IvyBridge M GT2",            { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        // { "Intel(R) HD Graphics Skylake ULT GT2",            { {"VW",8}, {"WGS",512}, {"WPT",1} } },
        // { "Iris",                                            { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        // { "Iris Pro",                                        { {"VW",1}, {"WGS",128}, {"WPT",2} } },
        { "default",                                         { {"VW",4}, {"WGS",256}, {"WPT",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        // { "Intel(R) Many Integrated Core Acceleration Card", { {"VW",2}, {"WGS",1024}, {"WPT",2} } },
        { "default",                                         { {"VW",2}, {"WGS",1024}, {"WPT",2} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        // { "GRID K520",                                       { {"VW",2}, {"WGS",64}, {"WPT",1} } },
        // { "GeForce GT 650M",                                 { {"VW",2}, {"WGS",1024}, {"WPT",1} } },
        // { "GeForce GTX 1070",                                { {"VW",1}, {"WGS",64}, {"WPT",4} } },
        { "GeForce GTX 1080",                                { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        // { "GeForce GTX 480",                                 { {"VW",2}, {"WGS",128}, {"WPT",1} } },
        // { "GeForce GTX 670",                                 { {"VW",2}, {"WGS",64}, {"WPT",1} } },
        // { "GeForce GTX 680",                                 { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        // { "GeForce GTX 750",                                 { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        // { "GeForce GTX 750 Ti",                              { {"VW",2}, {"WGS",64}, {"WPT",1} } },
        // { "GeForce GTX 980",                                 { {"VW",1}, {"WGS",1024}, {"WPT",1} } },
        // { "GeForce GTX TITAN",                               { {"VW",4}, {"WGS",256}, {"WPT",1} } },
        // { "GeForce GTX TITAN Black",                         { {"VW",4}, {"WGS",128}, {"WPT",4} } },
        // { "GeForce GTX TITAN X",                             { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        // { "TITAN X (Pascal)",                                { {"VW",4}, {"WGS",128}, {"WPT",1} } },
        // { "Tesla K20m",                                      { {"VW",4}, {"WGS",128}, {"WPT",1} } },
        // { "Tesla K40m",                                      { {"VW",4}, {"WGS",128}, {"WPT",1} } },
        { "default",                                         { {"VW",4}, {"WGS",1024}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW",4}, {"WGS",256}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XaxpyComplexSingle = {
  "Xaxpy", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW",2}, {"WGS",64}, {"WPT",8} } },
        { "ATI Radeon HD 6750M",                             { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "Ellesmere",                                       { {"VW",2}, {"WGS",256}, {"WPT",1} } },
        { "Fiji",                                            { {"VW",1}, {"WGS",128}, {"WPT",2} } },
        { "Hawaii",                                          { {"VW",1}, {"WGS",128}, {"WPT",2} } },
        { "Oland",                                           { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "Pitcairn",                                        { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "Tahiti",                                          { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "Tonga",                                           { {"VW",1}, {"WGS",256}, {"WPT",8} } },
        { "Turks",                                           { {"VW",2}, {"WGS",256}, {"WPT",1} } },
        { "default",                                         { {"VW",1}, {"WGS",128}, {"WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        { "default",                                         { {"VW",1}, {"WGS",256}, {"WPT",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"VW",4}, {"WGS",1024}, {"WPT",1} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW",4}, {"WGS",256}, {"WPT",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW",1}, {"WGS",1024}, {"WPT",2} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"VW",4}, {"WGS",1024}, {"WPT",1} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"VW",2}, {"WGS",1024}, {"WPT",1} } },
        { "default",                                         { {"VW",8}, {"WGS",1024}, {"WPT",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"VW",4}, {"WGS",64}, {"WPT",2} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"VW",4}, {"WGS",64}, {"WPT",1} } },
        { "Iris",                                            { {"VW",2}, {"WGS",128}, {"WPT",1} } },
        { "Iris Pro",                                        { {"VW",1}, {"WGS",256}, {"WPT",8} } },
        { "default",                                         { {"VW",4}, {"WGS",64}, {"WPT",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW",1}, {"WGS",1024}, {"WPT",1} } },
        { "default",                                         { {"VW",1}, {"WGS",1024}, {"WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"VW",1}, {"WGS",512}, {"WPT",1} } },
        { "GeForce GTX 1070",                                { {"VW",1}, {"WGS",64}, {"WPT",2} } },
        { "GeForce GTX 1080",                                { {"VW",2}, {"WGS",64}, {"WPT",1} } },
        { "GeForce GTX 480",                                 { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        { "GeForce GTX 670",                                 { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        { "GeForce GTX 680",                                 { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        { "GeForce GTX 750",                                 { {"VW",1}, {"WGS",512}, {"WPT",1} } },
        { "GeForce GTX 750 Ti",                              { {"VW",1}, {"WGS",512}, {"WPT",1} } },
        { "GeForce GTX 980",                                 { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "GeForce GTX TITAN",                               { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        { "GeForce GTX TITAN Black",                         { {"VW",1}, {"WGS",128}, {"WPT",2} } },
        { "GeForce GTX TITAN X",                             { {"VW",1}, {"WGS",512}, {"WPT",1} } },
        { "TITAN X (Pascal)",                                { {"VW",2}, {"WGS",512}, {"WPT",1} } },
        { "Tesla K20m",                                      { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "Tesla K40m",                                      { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "default",                                         { {"VW",1}, {"WGS",256}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW",1}, {"WGS",128}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XaxpyDouble = {
  "Xaxpy", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        { "Ellesmere",                                       { {"VW",2}, {"WGS",64}, {"WPT",4} } },
        { "Fiji",                                            { {"VW",2}, {"WGS",64}, {"WPT",4} } },
        { "Hawaii",                                          { {"VW",1}, {"WGS",64}, {"WPT",2} } },
        { "Oland",                                           { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "Pitcairn",                                        { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "Tahiti",                                          { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "Tonga",                                           { {"VW",1}, {"WGS",128}, {"WPT",4} } },
        { "default",                                         { {"VW",2}, {"WGS",64}, {"WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"VW",2}, {"WGS",128}, {"WPT",2} } },
        { "default",                                         { {"VW",2}, {"WGS",128}, {"WPT",2} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"VW",4}, {"WGS",64}, {"WPT",1} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW",1}, {"WGS",1024}, {"WPT",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW",8}, {"WGS",64}, {"WPT",1} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"VW",8}, {"WGS",256}, {"WPT",1} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"VW",8}, {"WGS",2048}, {"WPT",1} } },
        { "default",                                         { {"VW",8}, {"WGS",64}, {"WPT",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW",2}, {"WGS",512}, {"WPT",1} } },
        { "default",                                         { {"VW",2}, {"WGS",512}, {"WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "GeForce GTX 1070",                                { {"VW",1}, {"WGS",64}, {"WPT",8} } },
        { "GeForce GTX 1080",                                { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "GeForce GTX 480",                                 { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "GeForce GTX 670",                                 { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "GeForce GTX 680",                                 { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "GeForce GTX 750",                                 { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "GeForce GTX 750 Ti",                              { {"VW",1}, {"WGS",256}, {"WPT",2} } },
        { "GeForce GTX 980",                                 { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        { "GeForce GTX TITAN",                               { {"VW",2}, {"WGS",1024}, {"WPT",1} } },
        { "GeForce GTX TITAN Black",                         { {"VW",2}, {"WGS",128}, {"WPT",1} } },
        { "GeForce GTX TITAN X",                             { {"VW",1}, {"WGS",512}, {"WPT",1} } },
        { "TITAN X (Pascal)",                                { {"VW",2}, {"WGS",512}, {"WPT",1} } },
        { "Tesla K20m",                                      { {"VW",2}, {"WGS",128}, {"WPT",1} } },
        { "Tesla K40m",                                      { {"VW",2}, {"WGS",128}, {"WPT",1} } },
        { "default",                                         { {"VW",1}, {"WGS",128}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW",2}, {"WGS",256}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XaxpyComplexDouble = {
  "Xaxpy", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "Ellesmere",                                       { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "Fiji",                                            { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "Hawaii",                                          { {"VW",2}, {"WGS",64}, {"WPT",1} } },
        { "Oland",                                           { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        { "Pitcairn",                                        { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "Tahiti",                                          { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "Tonga",                                           { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "default",                                         { {"VW",1}, {"WGS",128}, {"WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"VW",1}, {"WGS",64}, {"WPT",8} } },
        { "default",                                         { {"VW",1}, {"WGS",64}, {"WPT",8} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"VW",4}, {"WGS",1024}, {"WPT",1} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW",8}, {"WGS",128}, {"WPT",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW",8}, {"WGS",512}, {"WPT",1} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"VW",8}, {"WGS",1024}, {"WPT",1} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        { "default",                                         { {"VW",4}, {"WGS",1024}, {"WPT",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW",1}, {"WGS",1024}, {"WPT",1} } },
        { "default",                                         { {"VW",1}, {"WGS",1024}, {"WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "GeForce GTX 1070",                                { {"VW",1}, {"WGS",64}, {"WPT",2} } },
        { "GeForce GTX 1080",                                { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        { "GeForce GTX 480",                                 { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "GeForce GTX 670",                                 { {"VW",1}, {"WGS",256}, {"WPT",1} } },
        { "GeForce GTX 680",                                 { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "GeForce GTX 750",                                 { {"VW",1}, {"WGS",1024}, {"WPT",1} } },
        { "GeForce GTX 750 Ti",                              { {"VW",1}, {"WGS",64}, {"WPT",2} } },
        { "GeForce GTX 980",                                 { {"VW",1}, {"WGS",1024}, {"WPT",1} } },
        { "GeForce GTX TITAN",                               { {"VW",1}, {"WGS",64}, {"WPT",4} } },
        { "GeForce GTX TITAN Black",                         { {"VW",1}, {"WGS",128}, {"WPT",4} } },
        { "GeForce GTX TITAN X",                             { {"VW",1}, {"WGS",1024}, {"WPT",1} } },
        { "TITAN X (Pascal)",                                { {"VW",1}, {"WGS",256}, {"WPT",2} } },
        { "Tesla K20m",                                      { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "Tesla K40m",                                      { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "default",                                         { {"VW",1}, {"WGS",64}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW",1}, {"WGS",256}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast

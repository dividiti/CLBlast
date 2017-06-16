
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xgemv_Fast' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XgemvFastHalf = {
  "XgemvFast", Precision::kHalf, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "Ellesmere",                                       { {"VW2",1}, {"WGS2",32}, {"WPT2",1} } },
        { "default",                                         { {"VW2",1}, {"WGS2",32}, {"WPT2",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        // { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"VW2",1}, {"WGS2",16}, {"WPT2",1} } },
        // { "Intel(R) HD Graphics Skylake ULT GT2",            { {"VW2",4}, {"WGS2",64}, {"WPT2",4} } },
        { "default",                                         { {"VW2",1}, {"WGS2",16}, {"WPT2",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW2",1}, {"WGS2",16}, {"WPT2",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastSingle = {
  "XgemvFast", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "AMD Radeon R9 M370X Compute Engine",              { {"VW2",1}, {"WGS2",128}, {"WPT2",1} } },
        // { "ATI Radeon HD 6750M",                             { {"VW2",2}, {"WGS2",64}, {"WPT2",2} } },
        // { "Ellesmere",                                       { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        // { "Fiji",                                            { {"VW2",1}, {"WGS2",64}, {"WPT2",2} } },
        // { "Hawaii",                                          { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        // { "Oland",                                           { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        // { "Pitcairn",                                        { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        // { "Tahiti",                                          { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        // { "Tonga",                                           { {"VW2",1}, {"WGS2",16}, {"WPT2",4} } },
        // { "Turks",                                           { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        // { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"VW2",1}, {"WGS2",32}, {"WPT2",4} } },
        // { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW2",4}, {"WGS2",128}, {"WPT2",4} } },
        // { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW2",1}, {"WGS2",64}, {"WPT2",4} } },
        // { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"VW2",2}, {"WGS2",16}, {"WPT2",4} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",4} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        // { "Intel(R) HD Graphics 530",                        { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        // { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"VW2",2}, {"WGS2",32}, {"WPT2",2} } },
        // { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"VW2",4}, {"WGS2",128}, {"WPT2",4} } },
        // { "Intel(R) HD Graphics IvyBridge M GT2",            { {"VW2",1}, {"WGS2",64}, {"WPT2",2} } },
        // { "Intel(R) HD Graphics Skylake ULT GT2",            { {"VW2",2}, {"WGS2",32}, {"WPT2",2} } },
        // { "Iris",                                            { {"VW2",1}, {"WGS2",128}, {"WPT2",2} } },
        // { "Iris Pro",                                        { {"VW2",4}, {"WGS2",64}, {"WPT2",4} } },
        { "default",                                         { {"VW2",2}, {"WGS2",256}, {"WPT2",2} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        // { "Intel(R) Many Integrated Core Acceleration Card", { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        // { "GRID K520",                                       { {"VW2",2}, {"WGS2",256}, {"WPT2",2} } },
        // { "GeForce GT 650M",                                 { {"VW2",2}, {"WGS2",32}, {"WPT2",2} } },
        // { "GeForce GTX 1070",                                { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        { "GeForce GTX 1080",                                { {"VW2",1}, {"WGS2",128}, {"WPT2",1} } },
        // { "GeForce GTX 480",                                 { {"VW2",1}, {"WGS2",128}, {"WPT2",1} } },
        // { "GeForce GTX 670",                                 { {"VW2",2}, {"WGS2",256}, {"WPT2",2} } },
        // { "GeForce GTX 680",                                 { {"VW2",1}, {"WGS2",128}, {"WPT2",1} } },
        // { "GeForce GTX 750",                                 { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        // { "GeForce GTX 750 Ti",                              { {"VW2",2}, {"WGS2",32}, {"WPT2",2} } },
        // { "GeForce GTX 980",                                 { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        // { "GeForce GTX TITAN",                               { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        // { "GeForce GTX TITAN Black",                         { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        // { "GeForce GTX TITAN X",                             { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        // { "TITAN X (Pascal)",                                { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        // { "Tesla K20m",                                      { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        // { "Tesla K40m",                                      { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        { "default",                                         { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastComplexSingle = {
  "XgemvFast", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW2",2}, {"WGS2",256}, {"WPT2",2} } },
        { "ATI Radeon HD 6750M",                             { {"VW2",1}, {"WGS2",128}, {"WPT2",1} } },
        { "Ellesmere",                                       { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "Fiji",                                            { {"VW2",1}, {"WGS2",16}, {"WPT2",1} } },
        { "Hawaii",                                          { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "Oland",                                           { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "Pitcairn",                                        { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "Tahiti",                                          { {"VW2",1}, {"WGS2",128}, {"WPT2",1} } },
        { "Tonga",                                           { {"VW2",2}, {"WGS2",32}, {"WPT2",2} } },
        { "Turks",                                           { {"VW2",1}, {"WGS2",16}, {"WPT2",1} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"VW2",2}, {"WGS2",64}, {"WPT2",4} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW2",1}, {"WGS2",128}, {"WPT2",2} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW2",4}, {"WGS2",64}, {"WPT2",4} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"VW2",4}, {"WGS2",16}, {"WPT2",4} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",2} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"VW2",2}, {"WGS2",128}, {"WPT2",2} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"VW2",1}, {"WGS2",32}, {"WPT2",2} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"VW2",2}, {"WGS2",128}, {"WPT2",2} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"VW2",1}, {"WGS2",32}, {"WPT2",4} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"VW2",1}, {"WGS2",32}, {"WPT2",1} } },
        { "Iris",                                            { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "Iris Pro",                                        { {"VW2",4}, {"WGS2",128}, {"WPT2",4} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        { "GeForce GTX 1070",                                { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "GeForce GTX 480",                                 { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "GeForce GTX 670",                                 { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "GeForce GTX 680",                                 { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastDouble = {
  "XgemvFast", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        { "Ellesmere",                                       { {"VW2",1}, {"WGS2",128}, {"WPT2",1} } },
        { "Fiji",                                            { {"VW2",1}, {"WGS2",32}, {"WPT2",1} } },
        { "Hawaii",                                          { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "Oland",                                           { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "Pitcairn",                                        { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "Tahiti",                                          { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "Tonga",                                           { {"VW2",2}, {"WGS2",32}, {"WPT2",2} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"VW2",1}, {"WGS2",64}, {"WPT2",4} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW2",4}, {"WGS2",128}, {"WPT2",4} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW2",1}, {"WGS2",64}, {"WPT2",4} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"VW2",1}, {"WGS2",16}, {"WPT2",4} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",4} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        { "GeForce GTX 1070",                                { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        { "GeForce GTX 1080",                                { {"VW2",1}, {"WGS2",32}, {"WPT2",2} } },
        { "GeForce GTX 480",                                 { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "GeForce GTX 670",                                 { {"VW2",1}, {"WGS2",128}, {"WPT2",1} } },
        { "GeForce GTX 680",                                 { {"VW2",1}, {"WGS2",128}, {"WPT2",1} } },
        { "GeForce GTX 750",                                 { {"VW2",2}, {"WGS2",256}, {"WPT2",2} } },
        { "GeForce GTX 750 Ti",                              { {"VW2",1}, {"WGS2",32}, {"WPT2",2} } },
        { "GeForce GTX 980",                                 { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "GeForce GTX TITAN",                               { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        { "GeForce GTX TITAN Black",                         { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        { "GeForce GTX TITAN X",                             { {"VW2",1}, {"WGS2",128}, {"WPT2",1} } },
        { "TITAN X (Pascal)",                                { {"VW2",1}, {"WGS2",32}, {"WPT2",1} } },
        { "Tesla K20m",                                      { {"VW2",1}, {"WGS2",128}, {"WPT2",1} } },
        { "Tesla K40m",                                      { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        { "default",                                         { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvFastComplexDouble = {
  "XgemvFast", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        { "Ellesmere",                                       { {"VW2",1}, {"WGS2",16}, {"WPT2",1} } },
        { "Fiji",                                            { {"VW2",1}, {"WGS2",16}, {"WPT2",1} } },
        { "Hawaii",                                          { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "Oland",                                           { {"VW2",1}, {"WGS2",256}, {"WPT2",1} } },
        { "Pitcairn",                                        { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "Tahiti",                                          { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "Tonga",                                           { {"VW2",1}, {"WGS2",32}, {"WPT2",1} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"VW2",4}, {"WGS2",32}, {"WPT2",4} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW2",2}, {"WGS2",64}, {"WPT2",4} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW2",4}, {"WGS2",64}, {"WPT2",4} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"VW2",1}, {"WGS2",16}, {"WPT2",2} } },
        { "default",                                         { {"VW2",4}, {"WGS2",64}, {"WPT2",4} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"VW2",1}, {"WGS2",128}, {"WPT2",1} } },
        { "GeForce GTX 480",                                 { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "GeForce GTX 670",                                 { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast

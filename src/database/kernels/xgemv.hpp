
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xgemv' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XgemvHalf = {
  "Xgemv", Precision::kHalf, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "Ellesmere",                                       { {"WGS1",256}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",256}, {"WPT1",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        // { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"WGS1",64}, {"WPT1",1} } },
        // { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",256}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WPT1",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvSingle = {
  "Xgemv", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",128}, {"WPT1",1} } },
        // { "ATI Radeon HD 6750M",                             { {"WGS1",32}, {"WPT1",1} } },
        // { "Ellesmere",                                       { {"WGS1",256}, {"WPT1",1} } },
        // { "Fiji",                                            { {"WGS1",128}, {"WPT1",1} } },
        // { "Hawaii",                                          { {"WGS1",128}, {"WPT1",1} } },
        // { "Oland",                                           { {"WGS1",128}, {"WPT1",1} } },
        // { "Pitcairn",                                        { {"WGS1",256}, {"WPT1",1} } },
        // { "Tahiti",                                          { {"WGS1",256}, {"WPT1",1} } },
        // { "Tonga",                                           { {"WGS1",128}, {"WPT1",2} } },
        // { "Turks",                                           { {"WGS1",32}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",128}, {"WPT1",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        // { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"WGS1",128}, {"WPT1",4} } },
        // { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",64}, {"WPT1",1} } },
        // { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"WGS1",64}, {"WPT1",4} } },
        // { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"WGS1",64}, {"WPT1",4} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",4} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        // { "Intel(R) HD Graphics 530",                        { {"WGS1",256}, {"WPT1",1} } },
        // { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"WGS1",64}, {"WPT1",1} } },
        // { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"WGS1",64}, {"WPT1",1} } },
        // { "Intel(R) HD Graphics IvyBridge M GT2",            { {"WGS1",256}, {"WPT1",1} } },
        // { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",32}, {"WPT1",1} } },
        // { "Iris",                                            { {"WGS1",64}, {"WPT1",2} } },
        // { "Iris Pro",                                        { {"WGS1",128}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",128}, {"WPT1",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        // { "Intel(R) Many Integrated Core Acceleration Card", { {"WGS1",64}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        // { "GRID K520",                                       { {"WGS1",256}, {"WPT1",1} } },
        // { "GeForce GT 650M",                                 { {"WGS1",256}, {"WPT1",1} } },
        // { "GeForce GTX 1070",                                { {"WGS1",128}, {"WPT1",1} } },
        { "GeForce GTX 1080",                                { {"WGS1",32}, {"WPT1",1} } },
        // { "GeForce GTX 480",                                 { {"WGS1",64}, {"WPT1",1} } },
        // { "GeForce GTX 670",                                 { {"WGS1",64}, {"WPT1",1} } },
        // { "GeForce GTX 680",                                 { {"WGS1",256}, {"WPT1",1} } },
        // { "GeForce GTX 750",                                 { {"WGS1",256}, {"WPT1",1} } },
        // { "GeForce GTX 750 Ti",                              { {"WGS1",32}, {"WPT1",1} } },
        // { "GeForce GTX 980",                                 { {"WGS1",128}, {"WPT1",1} } },
        // { "GeForce GTX TITAN",                               { {"WGS1",256}, {"WPT1",1} } },
        // { "GeForce GTX TITAN Black",                         { {"WGS1",256}, {"WPT1",1} } },
        // { "GeForce GTX TITAN X",                             { {"WGS1",256}, {"WPT1",1} } },
        // { "TITAN X (Pascal)",                                { {"WGS1",32}, {"WPT1",1} } },
        // { "Tesla K20m",                                      { {"WGS1",128}, {"WPT1",1} } },
        // { "Tesla K40m",                                      { {"WGS1",256}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",256}, {"WPT1",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",128}, {"WPT1",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvComplexSingle = {
  "Xgemv", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WPT1",1} } },
        // { "ATI Radeon HD 6750M",                             { {"WGS1",64}, {"WPT1",1} } },
        // { "Ellesmere",                                       { {"WGS1",32}, {"WPT1",1} } },
        // { "Fiji",                                            { {"WGS1",32}, {"WPT1",1} } },
        // { "Hawaii",                                          { {"WGS1",64}, {"WPT1",1} } },
        // { "Oland",                                           { {"WGS1",64}, {"WPT1",1} } },
        // { "Pitcairn",                                        { {"WGS1",64}, {"WPT1",1} } },
        // { "Tahiti",                                          { {"WGS1",64}, {"WPT1",1} } },
        // { "Tonga",                                           { {"WGS1",32}, {"WPT1",1} } },
        // { "Turks",                                           { {"WGS1",64}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        // { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"WGS1",32}, {"WPT1",4} } },
        // { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",128}, {"WPT1",1} } },
        // { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"WGS1",64}, {"WPT1",4} } },
        // { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"WGS1",64}, {"WPT1",4} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",2} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        // { "Intel(R) HD Graphics 530",                        { {"WGS1",64}, {"WPT1",1} } },
        // { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"WGS1",64}, {"WPT1",1} } },
        // { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"WGS1",128}, {"WPT1",1} } },
        // { "Intel(R) HD Graphics IvyBridge M GT2",            { {"WGS1",256}, {"WPT1",1} } },
        // { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",64}, {"WPT1",1} } },
        // { "Iris",                                            { {"WGS1",256}, {"WPT1",1} } },
        // { "Iris Pro",                                        { {"WGS1",64}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        // { "Intel(R) Many Integrated Core Acceleration Card", { {"WGS1",64}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        // { "GRID K520",                                       { {"WGS1",256}, {"WPT1",1} } },
        // { "GeForce GTX 1070",                                { {"WGS1",64}, {"WPT1",1} } },
        // { "GeForce GTX 1080",                                { {"WGS1",32}, {"WPT1",1} } },
        // { "GeForce GTX 480",                                 { {"WGS1",64}, {"WPT1",1} } },
        // { "GeForce GTX 670",                                 { {"WGS1",64}, {"WPT1",1} } },
        // { "GeForce GTX 680",                                 { {"WGS1",64}, {"WPT1",1} } },
        // { "GeForce GTX 750",                                 { {"WGS1",128}, {"WPT1",1} } },
        // { "GeForce GTX 750 Ti",                              { {"WGS1",32}, {"WPT1",1} } },
        // { "GeForce GTX TITAN",                               { {"WGS1",256}, {"WPT1",1} } },
        // { "GeForce GTX TITAN Black",                         { {"WGS1",32}, {"WPT1",1} } },
        // { "TITAN X (Pascal)",                                { {"WGS1",32}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WPT1",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvDouble = {
  "Xgemv", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WPT1",1} } },
        // { "Ellesmere",                                       { {"WGS1",32}, {"WPT1",1} } },
        // { "Fiji",                                            { {"WGS1",32}, {"WPT1",1} } },
        // { "Hawaii",                                          { {"WGS1",128}, {"WPT1",1} } },
        // { "Oland",                                           { {"WGS1",256}, {"WPT1",1} } },
        // { "Pitcairn",                                        { {"WGS1",256}, {"WPT1",1} } },
        // { "Tahiti",                                          { {"WGS1",256}, {"WPT1",1} } },
        // { "Tonga",                                           { {"WGS1",32}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",256}, {"WPT1",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        // { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"WGS1",64}, {"WPT1",4} } },
        // { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",64}, {"WPT1",2} } },
        // { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"WGS1",64}, {"WPT1",4} } },
        // { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"WGS1",64}, {"WPT1",4} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",4} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        // { "Intel(R) Many Integrated Core Acceleration Card", { {"WGS1",64}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        // { "GRID K520",                                       { {"WGS1",128}, {"WPT1",1} } },
        // { "GeForce GTX 1070",                                { {"WGS1",64}, {"WPT1",1} } },
        // { "GeForce GTX 1080",                                { {"WGS1",32}, {"WPT1",1} } },
        // { "GeForce GTX 480",                                 { {"WGS1",256}, {"WPT1",1} } },
        // { "GeForce GTX 670",                                 { {"WGS1",128}, {"WPT1",1} } },
        // { "GeForce GTX 680",                                 { {"WGS1",128}, {"WPT1",1} } },
        // { "GeForce GTX 750",                                 { {"WGS1",64}, {"WPT1",1} } },
        // { "GeForce GTX 750 Ti",                              { {"WGS1",32}, {"WPT1",1} } },
        // { "GeForce GTX 980",                                 { {"WGS1",64}, {"WPT1",1} } },
        // { "GeForce GTX TITAN",                               { {"WGS1",256}, {"WPT1",1} } },
        // { "GeForce GTX TITAN Black",                         { {"WGS1",32}, {"WPT1",1} } },
        // { "GeForce GTX TITAN X",                             { {"WGS1",64}, {"WPT1",1} } },
        // { "TITAN X (Pascal)",                                { {"WGS1",32}, {"WPT1",1} } },
        // { "Tesla K20m",                                      { {"WGS1",256}, {"WPT1",1} } },
        // { "Tesla K40m",                                      { {"WGS1",256}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",128}, {"WPT1",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",128}, {"WPT1",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemvComplexDouble = {
  "Xgemv", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WPT1",1} } },
        // { "Ellesmere",                                       { {"WGS1",32}, {"WPT1",1} } },
        // { "Fiji",                                            { {"WGS1",64}, {"WPT1",1} } },
        // { "Hawaii",                                          { {"WGS1",64}, {"WPT1",1} } },
        // { "Oland",                                           { {"WGS1",256}, {"WPT1",1} } },
        // { "Pitcairn",                                        { {"WGS1",256}, {"WPT1",1} } },
        // { "Tahiti",                                          { {"WGS1",256}, {"WPT1",1} } },
        // { "Tonga",                                           { {"WGS1",64}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        // { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"WGS1",64}, {"WPT1",4} } },
        // { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",64}, {"WPT1",1} } },
        // { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"WGS1",64}, {"WPT1",4} } },
        // { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"WGS1",32}, {"WPT1",4} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",4} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        // { "Intel(R) Many Integrated Core Acceleration Card", { {"WGS1",64}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        // { "GRID K520",                                       { {"WGS1",128}, {"WPT1",1} } },
        // { "GeForce GTX 480",                                 { {"WGS1",64}, {"WPT1",1} } },
        // { "GeForce GTX 670",                                 { {"WGS1",128}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",128}, {"WPT1",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WPT1",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast

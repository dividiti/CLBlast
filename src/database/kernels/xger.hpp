
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xger' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XgerHalf = {
  "Xger", Precision::kHalf, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "Ellesmere",                                       { {"WGS1",64}, {"WGS2",1}, {"WPT",2} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",1}, {"WPT",2} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        // { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"WGS1",256}, {"WGS2",1}, {"WPT",2} } },
        // { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",64}, {"WGS2",1}, {"WPT",4} } },
        { "default",                                         { {"WGS1",4}, {"WGS2",8}, {"WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WGS2",1}, {"WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgerSingle = {
  "Xger", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",256}, {"WGS2",1}, {"WPT",1} } },
        // { "ATI Radeon HD 6750M",                             { {"WGS1",16}, {"WGS2",16}, {"WPT",4} } },
        // { "Ellesmere",                                       { {"WGS1",64}, {"WGS2",4}, {"WPT",2} } },
        // { "Fiji",                                            { {"WGS1",256}, {"WGS2",1}, {"WPT",1} } },
        // { "Hawaii",                                          { {"WGS1",64}, {"WGS2",2}, {"WPT",1} } },
        // { "Oland",                                           { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        // { "Pitcairn",                                        { {"WGS1",64}, {"WGS2",1}, {"WPT",1} } },
        // { "Tahiti",                                          { {"WGS1",256}, {"WGS2",1}, {"WPT",1} } },
        // { "Tonga",                                           { {"WGS1",256}, {"WGS2",1}, {"WPT",2} } },
        // { "Turks",                                           { {"WGS1",64}, {"WGS2",4}, {"WPT",2} } },
        { "default",                                         { {"WGS1",16}, {"WGS2",16}, {"WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"WGS1",64}, {"WGS2",4}, {"WPT",4} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",4}, {"WPT",4} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        // { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"WGS1",32}, {"WGS2",4}, {"WPT",4} } },
        // { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",128}, {"WGS2",2}, {"WPT",4} } },
        // { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"WGS1",256}, {"WGS2",4}, {"WPT",4} } },
        // { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"WGS1",128}, {"WGS2",1}, {"WPT",4} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",8}, {"WPT",4} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        // { "Intel(R) HD Graphics 530",                        { {"WGS1",32}, {"WGS2",1}, {"WPT",2} } },
        // { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"WGS1",256}, {"WGS2",2}, {"WPT",2} } },
        // { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"WGS1",128}, {"WGS2",1}, {"WPT",2} } },
        // { "Intel(R) HD Graphics IvyBridge M GT2",            { {"WGS1",64}, {"WGS2",1}, {"WPT",4} } },
        // { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",32}, {"WGS2",4}, {"WPT",4} } },
        // { "Iris Pro",                                        { {"WGS1",64}, {"WGS2",1}, {"WPT",4} } },
        { "default",                                         { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        // { "GRID K520",                                       { {"WGS1",128}, {"WGS2",1}, {"WPT",2} } },
        // { "GeForce GT 650M",                                 { {"WGS1",32}, {"WGS2",16}, {"WPT",4} } },
        // { "GeForce GTX 1070",                                { {"WGS1",512}, {"WGS2",1}, {"WPT",1} } },
        { "GeForce GTX 1080",                                { {"WGS1",16}, {"WGS2",4}, {"WPT",1} } },
        // { "GeForce GTX 480",                                 { {"WGS1",256}, {"WGS2",1}, {"WPT",4} } },
        // { "GeForce GTX 670",                                 { {"WGS1",32}, {"WGS2",8}, {"WPT",2} } },
        // { "GeForce GTX 680",                                 { {"WGS1",128}, {"WGS2",1}, {"WPT",4} } },
        // { "GeForce GTX 750",                                 { {"WGS1",64}, {"WGS2",16}, {"WPT",4} } },
        // { "GeForce GTX 750 Ti",                              { {"WGS1",64}, {"WGS2",1}, {"WPT",2} } },
        // { "GeForce GTX TITAN",                               { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        // { "GeForce GTX TITAN Black",                         { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        // { "TITAN X (Pascal)",                                { {"WGS1",512}, {"WGS2",2}, {"WPT",1} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",1}, {"WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgerComplexSingle = {
  "Xger", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WGS2",4}, {"WPT",1} } },
        // { "ATI Radeon HD 6750M",                             { {"WGS1",16}, {"WGS2",16}, {"WPT",1} } },
        // { "Ellesmere",                                       { {"WGS1",16}, {"WGS2",8}, {"WPT",2} } },
        // { "Fiji",                                            { {"WGS1",128}, {"WGS2",2}, {"WPT",1} } },
        // { "Hawaii",                                          { {"WGS1",64}, {"WGS2",1}, {"WPT",2} } },
        // { "Oland",                                           { {"WGS1",4}, {"WGS2",8}, {"WPT",1} } },
        // { "Pitcairn",                                        { {"WGS1",128}, {"WGS2",2}, {"WPT",1} } },
        // { "Tahiti",                                          { {"WGS1",64}, {"WGS2",2}, {"WPT",1} } },
        // { "Tonga",                                           { {"WGS1",64}, {"WGS2",1}, {"WPT",1} } },
        // { "Turks",                                           { {"WGS1",128}, {"WGS2",2}, {"WPT",1} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",2}, {"WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        // { "Mali-T628",                                       { {"WGS1",128}, {"WGS2",1}, {"WPT",1} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",1}, {"WPT",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        // { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"WGS1",128}, {"WGS2",2}, {"WPT",4} } },
        // { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",256}, {"WGS2",1}, {"WPT",4} } },
        // { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"WGS1",256}, {"WGS2",2}, {"WPT",4} } },
        // { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"WGS1",512}, {"WGS2",4}, {"WPT",2} } },
        { "default",                                         { {"WGS1",256}, {"WGS2",2}, {"WPT",4} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        // { "Intel(R) HD Graphics 530",                        { {"WGS1",32}, {"WGS2",1}, {"WPT",2} } },
        // { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"WGS1",128}, {"WGS2",2}, {"WPT",1} } },
        // { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"WGS1",512}, {"WGS2",1}, {"WPT",1} } },
        // { "Intel(R) HD Graphics IvyBridge M GT2",            { {"WGS1",256}, {"WGS2",1}, {"WPT",2} } },
        // { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",16}, {"WGS2",1}, {"WPT",1} } },
        // { "Iris Pro",                                        { {"WGS1",16}, {"WGS2",2}, {"WPT",4} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",2}, {"WPT",2} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        // { "GRID K520",                                       { {"WGS1",64}, {"WGS2",4}, {"WPT",2} } },
        // { "GeForce GTX 1070",                                { {"WGS1",16}, {"WGS2",64}, {"WPT",2} } },
        // { "GeForce GTX 1080",                                { {"WGS1",32}, {"WGS2",2}, {"WPT",1} } },
        // { "GeForce GTX 480",                                 { {"WGS1",128}, {"WGS2",2}, {"WPT",2} } },
        // { "GeForce GTX 670",                                 { {"WGS1",16}, {"WGS2",32}, {"WPT",2} } },
        // { "GeForce GTX 680",                                 { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        // { "GeForce GTX 750",                                 { {"WGS1",32}, {"WGS2",16}, {"WPT",4} } },
        // { "GeForce GTX 750 Ti",                              { {"WGS1",32}, {"WGS2",8}, {"WPT",2} } },
        // { "GeForce GTX TITAN",                               { {"WGS1",16}, {"WGS2",16}, {"WPT",2} } },
        // { "GeForce GTX TITAN Black",                         { {"WGS1",16}, {"WGS2",16}, {"WPT",2} } },
        // { "TITAN X (Pascal)",                                { {"WGS1",32}, {"WGS2",2}, {"WPT",1} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",2}, {"WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WGS2",2}, {"WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgerDouble = {
  "Xger", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",32}, {"WGS2",4}, {"WPT",1} } },
        // { "Ellesmere",                                       { {"WGS1",64}, {"WGS2",1}, {"WPT",4} } },
        // { "Fiji",                                            { {"WGS1",256}, {"WGS2",1}, {"WPT",2} } },
        // { "Hawaii",                                          { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        // { "Oland",                                           { {"WGS1",128}, {"WGS2",1}, {"WPT",2} } },
        // { "Pitcairn",                                        { {"WGS1",64}, {"WGS2",1}, {"WPT",1} } },
        // { "Tahiti",                                          { {"WGS1",64}, {"WGS2",2}, {"WPT",1} } },
        // { "Tonga",                                           { {"WGS1",8}, {"WGS2",16}, {"WPT",2} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",2}, {"WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        // { "Mali-T628",                                       { {"WGS1",64}, {"WGS2",4}, {"WPT",1} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",4}, {"WPT",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        // { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"WGS1",256}, {"WGS2",1}, {"WPT",4} } },
        // { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",512}, {"WGS2",16}, {"WPT",1} } },
        // { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"WGS1",256}, {"WGS2",4}, {"WPT",4} } },
        // { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"WGS1",512}, {"WGS2",8}, {"WPT",2} } },
        { "default",                                         { {"WGS1",256}, {"WGS2",1}, {"WPT",4} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        // { "GRID K520",                                       { {"WGS1",128}, {"WGS2",8}, {"WPT",2} } },
        // { "GeForce GTX 1070",                                { {"WGS1",32}, {"WGS2",8}, {"WPT",1} } },
        // { "GeForce GTX 1080",                                { {"WGS1",32}, {"WGS2",2}, {"WPT",1} } },
        // { "GeForce GTX 480",                                 { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        // { "GeForce GTX 670",                                 { {"WGS1",32}, {"WGS2",32}, {"WPT",2} } },
        // { "GeForce GTX 680",                                 { {"WGS1",128}, {"WGS2",4}, {"WPT",2} } },
        // { "GeForce GTX 750",                                 { {"WGS1",256}, {"WGS2",2}, {"WPT",2} } },
        // { "GeForce GTX 750 Ti",                              { {"WGS1",32}, {"WGS2",16}, {"WPT",1} } },
        // { "GeForce GTX TITAN",                               { {"WGS1",16}, {"WGS2",8}, {"WPT",2} } },
        // { "GeForce GTX TITAN Black",                         { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        // { "TITAN X (Pascal)",                                { {"WGS1",32}, {"WGS2",2}, {"WPT",1} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",1}, {"WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",128}, {"WGS2",1}, {"WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgerComplexDouble = {
  "Xger", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WGS2",1}, {"WPT",1} } },
        // { "Ellesmere",                                       { {"WGS1",8}, {"WGS2",16}, {"WPT",1} } },
        // { "Fiji",                                            { {"WGS1",64}, {"WGS2",4}, {"WPT",2} } },
        // { "Hawaii",                                          { {"WGS1",128}, {"WGS2",1}, {"WPT",1} } },
        // { "Oland",                                           { {"WGS1",16}, {"WGS2",16}, {"WPT",2} } },
        // { "Pitcairn",                                        { {"WGS1",64}, {"WGS2",4}, {"WPT",1} } },
        // { "Tahiti",                                          { {"WGS1",32}, {"WGS2",4}, {"WPT",1} } },
        // { "Tonga",                                           { {"WGS1",16}, {"WGS2",4}, {"WPT",1} } },
        { "default",                                         { {"WGS1",32}, {"WGS2",4}, {"WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        // { "Mali-T628",                                       { {"WGS1",64}, {"WGS2",2}, {"WPT",4} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",2}, {"WPT",4} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        // { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"WGS1",128}, {"WGS2",4}, {"WPT",4} } },
        // { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",512}, {"WGS2",4}, {"WPT",2} } },
        // { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"WGS1",512}, {"WGS2",2}, {"WPT",2} } },
        // { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"WGS1",256}, {"WGS2",1}, {"WPT",2} } },
        { "default",                                         { {"WGS1",256}, {"WGS2",1}, {"WPT",2} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        // { "GRID K520",                                       { {"WGS1",16}, {"WGS2",8}, {"WPT",2} } },
        // { "GeForce GTX 1070",                                { {"WGS1",8}, {"WGS2",128}, {"WPT",1} } },
        // { "GeForce GTX 1080",                                { {"WGS1",8}, {"WGS2",4}, {"WPT",1} } },
        // { "GeForce GTX 480",                                 { {"WGS1",64}, {"WGS2",2}, {"WPT",2} } },
        // { "GeForce GTX 670",                                 { {"WGS1",8}, {"WGS2",16}, {"WPT",2} } },
        // { "GeForce GTX 680",                                 { {"WGS1",8}, {"WGS2",16}, {"WPT",1} } },
        // { "GeForce GTX 750",                                 { {"WGS1",8}, {"WGS2",32}, {"WPT",4} } },
        // { "GeForce GTX 750 Ti",                              { {"WGS1",32}, {"WGS2",8}, {"WPT",2} } },
        // { "GeForce GTX TITAN",                               { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        // { "GeForce GTX TITAN Black",                         { {"WGS1",16}, {"WGS2",16}, {"WPT",2} } },
        // { "TITAN X (Pascal)",                                { {"WGS1",4}, {"WGS2",8}, {"WPT",1} } },
        { "default",                                         { {"WGS1",16}, {"WGS2",8}, {"WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WGS2",2}, {"WPT",2} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast


// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xgemm' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {
// =================================================================================================

const Database::DatabaseEntry XgemmHalf = {
  "Xgemm", Precision::kHalf, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "Ellesmere",                                       { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        // { "Intel(R) HD Graphics Skylake ULT GT2",            { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmSingle = {
  "Xgemm", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        // { "AMD Radeon R9 M370X Compute Engine",              { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",8} } },
        // { "ATI Radeon HD 6750M",                             { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",128}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",8}, {"VWN",8} } },
        // { "Ellesmere",                                       { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        // { "Fiji",                                            { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
        // { "Hawaii",                                          { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",32}, {"MWG",128}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",4}, {"VWN",2} } },
        // { "Oland",                                           { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
        // { "Pitcairn",                                        { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        // { "Tahiti",                                          { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",32}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",1} } },
        // { "Tonga",                                           { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",32}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",8} } },
        // { "Turks",                                           { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",16}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",8}, {"VWN",1} } },
        { "default",                                         { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",16}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",8}, {"VWN",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        // { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",128}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",1}, {"VWN",8} } },
        // { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"KWG",32}, {"KWI",8}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",2} } },
        // { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",8} } },
        // { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",8} } },
        // { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"KWG",32}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",32}, {"NWG",64}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        // { "Intel(R) HD Graphics 530",                        { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",4}, {"VWN",2} } },
        // { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"KWG",32}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",4}, {"VWN",2} } },
        // { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",4} } },
        // { "Intel(R) HD Graphics IvyBridge M GT2",            { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
        // { "Intel(R) HD Graphics Skylake ULT GT2",            { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
        // { "Iris",                                            { {"KWG",16}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",4}, {"VWN",1} } },
        // { "Iris Pro",                                        { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",32}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        // { "Intel(R) Many Integrated Core Acceleration Card", { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        // { "GRID K520",                                       { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",4} } },
        // { "GeForce GT 650M",                                 { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",32}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",2} } },
        // { "GeForce GTX 1070",                                { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",128}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",4}, {"VWN",1} } },
        { "GeForce GTX 1080",                                { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",4}, {"VWN",8} } },
        // { "GeForce GTX 480",                                 { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",2} } },
        // { "GeForce GTX 670",                                 { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
        // { "GeForce GTX 680",                                 { {"KWG",32}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",2} } },
        // { "GeForce GTX 750",                                 { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",1}, {"VWN",2} } },
        // { "GeForce GTX 750 Ti",                              { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",128}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",1}, {"VWM",8}, {"VWN",2} } },
        // { "GeForce GTX 980",                                 { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",4}, {"VWN",8} } },
        // { "GeForce GTX TITAN",                               { {"KWG",16}, {"KWI",8}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",2} } },
        // { "GeForce GTX TITAN Black",                         { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",4}, {"VWN",1} } },
        // { "GeForce GTX TITAN X",                             { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",128}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",4}, {"VWN",8} } },
        // { "TITAN X (Pascal)",                                { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",1} } },
        // { "Tesla K20m",                                      { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
        // { "Tesla K40m",                                      { {"KWG",16}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmComplexSingle = {
  "Xgemm", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",8} } },
        { "ATI Radeon HD 6750M",                             { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
        { "Ellesmere",                                       { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
        { "Fiji",                                            { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "Hawaii",                                          { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",32}, {"SA",1}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "Oland",                                           { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",32}, {"NWG",128}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",1}, {"VWM",2}, {"VWN",4} } },
        { "Pitcairn",                                        { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",4}, {"VWN",2} } },
        { "Tahiti",                                          { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",32}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",1}, {"VWM",2}, {"VWN",1} } },
        { "Tonga",                                           { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",32}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",1} } },
        { "Turks",                                           { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",1} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",1}, {"VWM",8}, {"VWN",1} } },
        { "default",                                         { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",1}, {"VWM",8}, {"VWN",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",1}, {"VWM",1}, {"VWN",2} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",32}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",1}, {"VWN",4} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"KWG",16}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",1} } },
        { "Intel(R) HD Graphics 5500 BroadWell U-Processor GT2", { {"KWG",16}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"KWG",32}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",1} } },
        { "Intel(R) HD Graphics IvyBridge M GT2",            { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",2} } },
        { "Iris",                                            { {"KWG",32}, {"KWI",8}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "Iris Pro",                                        { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",1}, {"VWN",1} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",2} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",128}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",128}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"KWG",16}, {"KWI",8}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",128}, {"SA",1}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
        { "GeForce GTX 1070",                                { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",4} } },
        { "GeForce GTX 1080",                                { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "GeForce GTX 480",                                 { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",128}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",2} } },
        { "GeForce GTX 670",                                 { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",1}, {"VWN",1} } },
        { "GeForce GTX 680",                                 { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",32}, {"NWG",128}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",2} } },
        { "GeForce GTX 750",                                 { {"KWG",16}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",2} } },
        { "GeForce GTX 750 Ti",                              { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "GeForce GTX 980",                                 { {"KWG",32}, {"KWI",8}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",1} } },
        { "GeForce GTX TITAN",                               { {"KWG",16}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "GeForce GTX TITAN Black",                         { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "GeForce GTX TITAN X",                             { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",1}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",1}, {"VWN",4} } },
        { "TITAN X (Pascal)",                                { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
        { "Tesla K20m",                                      { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
        { "Tesla K40m",                                      { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",64}, {"SA",0}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",16}, {"NDIMB",32}, {"NDIMC",32}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmDouble = {
  "Xgemm", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
        { "Ellesmere",                                       { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",2} } },
        { "Fiji",                                            { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",2} } },
        { "Hawaii",                                          { {"KWG",16}, {"KWI",8}, {"MDIMA",32}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
        { "Oland",                                           { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",16}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",1}, {"VWN",1} } },
        { "Pitcairn",                                        { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "Tahiti",                                          { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",16}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",1}, {"VWM",1}, {"VWN",4} } },
        { "Tonga",                                           { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",2} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",2} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",16}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",8}, {"VWN",2} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",16}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",8}, {"VWN",2} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",8} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",1}, {"VWM",2}, {"VWN",8} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",128}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",8} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",1}, {"VWN",8} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"KWG",32}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
        { "default",                                         { {"KWG",32}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",16}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",1}, {"VWM",2}, {"VWN",2} } },
        { "GeForce GTX 1070",                                { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",8} } },
        { "GeForce GTX 1080",                                { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
        { "GeForce GTX 480",                                 { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",64}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "GeForce GTX 670",                                 { {"KWG",32}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",32}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "GeForce GTX 680",                                 { {"KWG",32}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",32}, {"NWG",128}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",1}, {"VWM",2}, {"VWN",4} } },
        { "GeForce GTX 750",                                 { {"KWG",32}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",32}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",1}, {"VWM",2}, {"VWN",1} } },
        { "GeForce GTX 750 Ti",                              { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",2} } },
        { "GeForce GTX 980",                                 { {"KWG",32}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",32}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
        { "GeForce GTX TITAN",                               { {"KWG",16}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",32}, {"NWG",128}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",2} } },
        { "GeForce GTX TITAN Black",                         { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",16}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "GeForce GTX TITAN X",                             { {"KWG",16}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",16}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "TITAN X (Pascal)",                                { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "Tesla K20m",                                      { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "Tesla K40m",                                      { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",32}, {"NWG",128}, {"SA",1}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",4} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",4}, {"VWN",4} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry XgemmComplexDouble = {
  "Xgemm", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"KWG",32}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",2} } },
        { "Ellesmere",                                       { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",16}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "Fiji",                                            { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",16}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "Hawaii",                                          { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",32}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "Oland",                                           { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",32}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
        { "Pitcairn",                                        { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",32}, {"NWG",32}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "Tahiti",                                          { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",16}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",16}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "Tonga",                                           { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",16}, {"SA",1}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",1}, {"VWN",1} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",16}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",8}, {"VWN",1} } },
        { "default",                                         { {"KWG",16}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",8}, {"VWN",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz",       { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",128}, {"SA",0}, {"SB",1}, {"STRM",0}, {"STRN",1}, {"VWM",2}, {"VWN",1} } },
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",32}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",0}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",2}, {"VWN",4} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",1}, {"VWM",1}, {"VWN",8} } },
        { "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz",        { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",128}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"KWG",32}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",32}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",16}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"KWG",32}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",64}, {"SA",1}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",1}, {"VWN",1} } },
        { "GeForce GTX 1070",                                { {"KWG",32}, {"KWI",8}, {"MDIMA",32}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",1}, {"VWM",1}, {"VWN",4} } },
        { "GeForce GTX 1080",                                { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",16}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",2} } },
        { "GeForce GTX 480",                                 { {"KWG",16}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "GeForce GTX 670",                                 { {"KWG",32}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",32}, {"NWG",64}, {"SA",1}, {"SB",0}, {"STRM",0}, {"STRN",1}, {"VWM",1}, {"VWN",2} } },
        { "GeForce GTX 680",                                 { {"KWG",16}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",64}, {"NDIMB",16}, {"NDIMC",32}, {"NWG",32}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "GeForce GTX 750",                                 { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",32}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",4} } },
        { "GeForce GTX 750 Ti",                              { {"KWG",32}, {"KWI",2}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",16}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "GeForce GTX 980",                                 { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",16}, {"NWG",128}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",1}, {"VWM",2}, {"VWN",2} } },
        { "GeForce GTX TITAN Black",                         { {"KWG",16}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",8}, {"NWG",32}, {"SA",0}, {"SB",1}, {"STRM",1}, {"STRN",1}, {"VWM",1}, {"VWN",1} } },
        { "GeForce GTX TITAN X",                             { {"KWG",32}, {"KWI",8}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",128}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "TITAN X (Pascal)",                                { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",16}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",16}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "Tesla K20m",                                      { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",64}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "Tesla K40m",                                      { {"KWG",16}, {"KWI",8}, {"MDIMA",8}, {"MDIMC",8}, {"MWG",32}, {"NDIMB",32}, {"NDIMC",16}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",1}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",16}, {"MDIMC",16}, {"MWG",32}, {"NDIMB",16}, {"NDIMC",16}, {"NWG",32}, {"SA",0}, {"SB",0}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"KWG",32}, {"KWI",2}, {"MDIMA",32}, {"MDIMC",32}, {"MWG",32}, {"NDIMB",8}, {"NDIMC",8}, {"NWG",32}, {"SA",1}, {"SB",1}, {"STRM",0}, {"STRN",0}, {"VWM",1}, {"VWN",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace database
} // namespace clblast

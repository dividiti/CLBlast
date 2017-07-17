
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the common (test) utility functions.
//
// =================================================================================================

#include "utilities/utilities.hpp"

#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

namespace clblast {
// =================================================================================================

// Returns a scalar with a default value
template <typename T> T GetScalar() { return static_cast<T>(2.0); }
template float GetScalar<float>();
template double GetScalar<double>();
template <> half GetScalar() { return FloatToHalf(2.0f); }
template <> float2 GetScalar() { return {2.0f, 0.5f}; }
template <> double2 GetScalar() { return {2.0, 0.5}; }

// Returns a scalar of value 0
template <typename T> T ConstantZero() { return static_cast<T>(0.0); }
template float ConstantZero<float>();
template double ConstantZero<double>();
template <> half ConstantZero() { return FloatToHalf(0.0f); }
template <> float2 ConstantZero() { return {0.0f, 0.0f}; }
template <> double2 ConstantZero() { return {0.0, 0.0}; }

// Returns a scalar of value 1
template <typename T> T ConstantOne() { return static_cast<T>(1.0); }
template float ConstantOne<float>();
template double ConstantOne<double>();
template <> half ConstantOne() { return FloatToHalf(1.0f); }
template <> float2 ConstantOne() { return {1.0f, 0.0f}; }
template <> double2 ConstantOne() { return {1.0, 0.0}; }

// Returns a scalar of value -1
template <typename T> T ConstantNegOne() { return static_cast<T>(-1.0); }
template float ConstantNegOne<float>();
template double ConstantNegOne<double>();
template <> half ConstantNegOne() { return FloatToHalf(-1.0f); }
template <> float2 ConstantNegOne() { return {-1.0f, 0.0f}; }
template <> double2 ConstantNegOne() { return {-1.0, 0.0}; }

// Returns a scalar of some value
template <typename T> T Constant(const double val) { return static_cast<T>(val); }
template float Constant<float>(const double);
template double Constant<double>(const double);
template <> half Constant(const double val) { return FloatToHalf(static_cast<float>(val)); }
template <> float2 Constant(const double val) { return {static_cast<float>(val), 0.0f}; }
template <> double2 Constant(const double val) { return {val, 0.0}; }

// Returns a small scalar value just larger than 0
template <typename T> T SmallConstant() { return static_cast<T>(1e-4); }
template float SmallConstant<float>();
template double SmallConstant<double>();
template <> half SmallConstant() { return FloatToHalf(1e-4f); }
template <> float2 SmallConstant() { return {1e-4f, 0.0f}; }
template <> double2 SmallConstant() { return {1e-4, 0.0}; }

// Returns the absolute value of a scalar (modulus in case of a complex number)
template <typename T> typename BaseType<T>::Type AbsoluteValue(const T value) { return std::fabs(value); }
template float AbsoluteValue<float>(const float);
template double AbsoluteValue<double>(const double);
template <> half AbsoluteValue(const half value) { return FloatToHalf(std::fabs(HalfToFloat(value))); }
template <> float AbsoluteValue(const float2 value) {
  if (value.real() == 0.0f && value.imag() == 0.0f) { return 0.0f; }
  return std::sqrt(value.real() * value.real() + value.imag() * value.imag());
}
template <> double AbsoluteValue(const double2 value) {
  if (value.real() == 0.0 && value.imag() == 0.0) { return 0.0; }
  return std::sqrt(value.real() * value.real() + value.imag() * value.imag());
}

// Returns whether a scalar is close to zero
template <typename T> bool IsCloseToZero(const T value) { return (value > -SmallConstant<T>()) && (value < SmallConstant<T>()); }
template bool IsCloseToZero<float>(const float);
template bool IsCloseToZero<double>(const double);
template <> bool IsCloseToZero(const half value) { return IsCloseToZero(HalfToFloat(value)); }
template <> bool IsCloseToZero(const float2 value) { return IsCloseToZero(value.real()) || IsCloseToZero(value.imag()); }
template <> bool IsCloseToZero(const double2 value) { return IsCloseToZero(value.real()) || IsCloseToZero(value.imag()); }

// =================================================================================================

// Implements the string conversion using std::to_string if possible
template <typename T>
std::string ToString(T value) {
  return std::to_string(value);
}
template std::string ToString<int>(int value);
template std::string ToString<size_t>(size_t value);
template <>
std::string ToString(float value) {
  std::ostringstream result;
  result << std::fixed << std::setprecision(2) << value;
  return result.str();
}
template <>
std::string ToString(double value) {
  std::ostringstream result;
  result << std::fixed << std::setprecision(2) << value;
  return result.str();
}

// If not possible directly: special cases for complex data-types
template <>
std::string ToString(float2 value) {
  return ToString(value.real())+"+"+ToString(value.imag())+"i";
}
template <>
std::string ToString(double2 value) {
  return ToString(value.real())+"+"+ToString(value.imag())+"i";
}

// If not possible directly: special case for half-precision
template <>
std::string ToString(half value) {
  return std::to_string(HalfToFloat(value));
}

// If not possible directly: special cases for CLBlast data-types
template <>
std::string ToString(Layout value) {
  switch(value) {
    case Layout::kRowMajor: return ToString(static_cast<int>(value))+" (row-major)";
    case Layout::kColMajor: return ToString(static_cast<int>(value))+" (col-major)";
  }
}
template <>
std::string ToString(Transpose value) {
  switch(value) {
    case Transpose::kNo: return ToString(static_cast<int>(value))+" (regular)";
    case Transpose::kYes: return ToString(static_cast<int>(value))+" (transposed)";
    case Transpose::kConjugate: return ToString(static_cast<int>(value))+" (conjugate)";
  }
}
template <>
std::string ToString(Side value) {
  switch(value) {
    case Side::kLeft: return ToString(static_cast<int>(value))+" (left)";
    case Side::kRight: return ToString(static_cast<int>(value))+" (right)";
  }
}
template <>
std::string ToString(Triangle value) {
  switch(value) {
    case Triangle::kUpper: return ToString(static_cast<int>(value))+" (upper)";
    case Triangle::kLower: return ToString(static_cast<int>(value))+" (lower)";
  }
}
template <>
std::string ToString(Diagonal value) {
  switch(value) {
    case Diagonal::kUnit: return ToString(static_cast<int>(value))+" (unit)";
    case Diagonal::kNonUnit: return ToString(static_cast<int>(value))+" (non-unit)";
  }
}
template <>
std::string ToString(Precision value) {
  switch(value) {
    case Precision::kHalf: return ToString(static_cast<int>(value))+" (half)";
    case Precision::kSingle: return ToString(static_cast<int>(value))+" (single)";
    case Precision::kDouble: return ToString(static_cast<int>(value))+" (double)";
    case Precision::kComplexSingle: return ToString(static_cast<int>(value))+" (complex-single)";
    case Precision::kComplexDouble: return ToString(static_cast<int>(value))+" (complex-double)";
    case Precision::kAny: return ToString(static_cast<int>(value))+" (any)";
  }
}
template <>
std::string ToString(StatusCode value) {
  return std::to_string(static_cast<int>(value));
}

// =================================================================================================

// Retrieves the command-line arguments in a C++ fashion. Also adds command-line arguments from
// pre-defined environmental variables
std::vector<std::string> RetrieveCommandLineArguments(int argc, char *argv[]) {

  // Regular command-line arguments
  auto command_line_args = std::vector<std::string>();
  for (auto i=0; i<argc; ++i) {
    command_line_args.push_back(std::string{argv[i]});
  }

  // Extra CLBlast arguments
  const auto extra_args = ConvertArgument(std::getenv("CLBLAST_ARGUMENTS"), std::string{""});
  std::stringstream extra_args_stream;
  extra_args_stream.str(extra_args);
  std::string extra_arg;
  while (std::getline(extra_args_stream, extra_arg, ' ')) {
    command_line_args.push_back(extra_arg);
  }
  return command_line_args;
}

// Helper for the below function to convert the argument to the value type. Adds specialization for
// complex data-types. Note that complex arguments are accepted as regular values and are copied to
// both the real and imaginary parts.
template <typename T>
T ConvertArgument(const char* value) {
  return static_cast<T>(std::stoi(value));
}
template size_t ConvertArgument(const char* value);

template <> std::string ConvertArgument(const char* value) {
  return std::string{value};
}
template <> half ConvertArgument(const char* value) {
  return FloatToHalf(static_cast<float>(std::stod(value)));
}
template <> float ConvertArgument(const char* value) {
  return static_cast<float>(std::stod(value));
}
template <> double ConvertArgument(const char* value) {
  return static_cast<double>(std::stod(value));
}
template <> float2 ConvertArgument(const char* value) {
  auto val = static_cast<float>(std::stod(value));
  return float2{val, val};
}
template <> double2 ConvertArgument(const char* value) {
  auto val = static_cast<double>(std::stod(value));
  return double2{val, val};
}

// Variant of "ConvertArgument" with default values
template <typename T>
T ConvertArgument(const char* value, T default_value) {

  if (value) { return ConvertArgument<T>(value); }
  return default_value;
}
template size_t ConvertArgument(const char* value, size_t default_value);
template std::string ConvertArgument(const char* value, std::string default_value);

// This function matches patterns in the form of "-option value" or "--option value". It returns a
// default value in case the option is not found in the argument string.
template <typename T>
T GetArgument(const std::vector<std::string> &arguments, std::string &help,
              const std::string &option, const T default_value) {

  // Parses the argument. Note that this supports both the given option (e.g. -device) and one with
  // an extra dash in front (e.g. --device).
  auto return_value = static_cast<T>(default_value);
  for (auto c=size_t{0}; c<arguments.size(); ++c) {
    auto item = arguments[c];
    if (item.compare("-"+option) == 0 || item.compare("--"+option) == 0) {
      ++c;
      return_value = ConvertArgument<T>(arguments[c].c_str());
      break;
    }
  }

  // Updates the help message and returns
  help += "    -"+option+" "+ToString(return_value)+" ";
  help += (return_value == default_value) ? "[=default]\n" : "\n";
  return return_value;
}

// Compiles the above function
template int GetArgument<int>(const std::vector<std::string>&, std::string&, const std::string&, const int);
template size_t GetArgument<size_t>(const std::vector<std::string>&, std::string&, const std::string&, const size_t);
template half GetArgument<half>(const std::vector<std::string>&, std::string&, const std::string&, const half);
template float GetArgument<float>(const std::vector<std::string>&, std::string&, const std::string&, const float);
template double GetArgument<double>(const std::vector<std::string>&, std::string&, const std::string&, const double);
template float2 GetArgument<float2>(const std::vector<std::string>&, std::string&, const std::string&, const float2);
template double2 GetArgument<double2>(const std::vector<std::string>&, std::string&, const std::string&, const double2);
template Layout GetArgument<Layout>(const std::vector<std::string>&, std::string&, const std::string&, const Layout);
template Transpose GetArgument<Transpose>(const std::vector<std::string>&, std::string&, const std::string&, const Transpose);
template Side GetArgument<Side>(const std::vector<std::string>&, std::string&, const std::string&, const Side);
template Triangle GetArgument<Triangle>(const std::vector<std::string>&, std::string&, const std::string&, const Triangle);
template Diagonal GetArgument<Diagonal>(const std::vector<std::string>&, std::string&, const std::string&, const Diagonal);
template Precision GetArgument<Precision>(const std::vector<std::string>&, std::string&, const std::string&, const Precision);


std::string GetStringArgument(const std::vector<std::string>&arguments, std::string&help, const std::string&option, const std::string&default_value)
{
  // Parses the argument. Note that this supports both the given option (e.g. -device) and one with
  // an extra dash in front (e.g. --device).
  auto return_value = (default_value);
  for (auto c=size_t{0}; c<arguments.size(); ++c) {
    auto item = arguments[c];
    if (item.compare("-"+option) == 0 || item.compare("--"+option) == 0) {
      ++c;
      return_value = arguments[c].c_str();
      break;
    }
  }

  // Updates the help message and returns
  help += "    -"+option+" "+return_value+" ";
  help += (return_value == default_value) ? "[=default]\n" : "\n";
  return return_value;
}
// =================================================================================================

// Returns only the precision argument
Precision GetPrecision(const std::vector<std::string> &arguments, const Precision default_precision) {
  auto dummy = std::string{};
  return GetArgument(arguments, dummy, kArgPrecision, default_precision);
}

// =================================================================================================

// Checks whether an argument is given. Returns true or false.
bool CheckArgument(const std::vector<std::string> &arguments, std::string &help,
                   const std::string &option) {

  // Parses the argument. Note that this supports both the given option (e.g. -device) and one with
  // an extra dash in front (e.g. --device).
  auto return_value = false;
  for (auto c=size_t{0}; c<arguments.size(); ++c) {
    auto item = arguments[c];
    if (item.compare("-"+option) == 0 || item.compare("--"+option) == 0) {
      ++c;
      return_value = true;
    }
  }

  // Updates the help message and returns
  help += "    -"+option+" ";
  help += (return_value) ? "[true]\n" : "[false]\n";
  return return_value;
}

// =================================================================================================

// Returns a random seed. This used to be implemented using 'std::random_device', but that doesn't
// always work. The chrono-timers are more reliable in that sense, but perhaps less random.
unsigned int GetRandomSeed() {
  return static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
}

// Create a random number generator and populates a vector with samples from a random distribution
template <typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
  for (auto &element: vector) { element = static_cast<T>(dist(mt)); }
}
template void PopulateVector<float>(std::vector<float>&, std::mt19937&, std::uniform_real_distribution<double>&);
template void PopulateVector<double>(std::vector<double>&, std::mt19937&, std::uniform_real_distribution<double>&);

// Specialized versions of the above for complex data-types
template <>
void PopulateVector(std::vector<float2> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
  for (auto &element: vector) {
    element.real(static_cast<float>(dist(mt)));
    element.imag(static_cast<float>(dist(mt)));
  }
}
template <>
void PopulateVector(std::vector<double2> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
  for (auto &element: vector) { element.real(dist(mt)); element.imag(dist(mt)); }
}

// Specialized versions of the above for half-precision
template <>
void PopulateVector(std::vector<half> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
  for (auto &element: vector) { element = FloatToHalf(static_cast<float>(dist(mt))); }
}

// =================================================================================================

template <typename T, typename U>
void DeviceToHost(const Arguments<U> &args, Buffers<T> &buffers, BuffersHost<T> &buffers_host,
                  Queue &queue, const std::vector<std::string> &names) {
  for (auto &name: names) {
    if (name == kBufVecX) {buffers_host.x_vec = std::vector<T>(args.x_size, static_cast<T>(0)); buffers.x_vec.Read(queue, args.x_size, buffers_host.x_vec); }
    else if (name == kBufVecY) { buffers_host.y_vec = std::vector<T>(args.y_size, static_cast<T>(0)); buffers.y_vec.Read(queue, args.y_size, buffers_host.y_vec); }
    else if (name == kBufMatA) { buffers_host.a_mat = std::vector<T>(args.a_size, static_cast<T>(0)); buffers.a_mat.Read(queue, args.a_size, buffers_host.a_mat); }
    else if (name == kBufMatB) { buffers_host.b_mat = std::vector<T>(args.b_size, static_cast<T>(0)); buffers.b_mat.Read(queue, args.b_size, buffers_host.b_mat); }
    else if (name == kBufMatC) { buffers_host.c_mat = std::vector<T>(args.c_size, static_cast<T>(0)); buffers.c_mat.Read(queue, args.c_size, buffers_host.c_mat); }
    else if (name == kBufMatAP) { buffers_host.ap_mat = std::vector<T>(args.ap_size, static_cast<T>(0)); buffers.ap_mat.Read(queue, args.ap_size, buffers_host.ap_mat); }
    else if (name == kBufScalar) { buffers_host.scalar = std::vector<T>(args.scalar_size, static_cast<T>(0)); buffers.scalar.Read(queue, args.scalar_size, buffers_host.scalar); }
    else { throw std::runtime_error("Invalid buffer name"); }
  }
}

template <typename T, typename U>
void HostToDevice(const Arguments<U> &args, Buffers<T> &buffers, BuffersHost<T> &buffers_host,
                  Queue &queue, const std::vector<std::string> &names) {
  for (auto &name: names) {
    if (name == kBufVecX) { buffers.x_vec.Write(queue, args.x_size, buffers_host.x_vec); }
    else if (name == kBufVecY) { buffers.y_vec.Write(queue, args.y_size, buffers_host.y_vec); }
    else if (name == kBufMatA) { buffers.a_mat.Write(queue, args.a_size, buffers_host.a_mat); }
    else if (name == kBufMatB) { buffers.b_mat.Write(queue, args.b_size, buffers_host.b_mat); }
    else if (name == kBufMatC) { buffers.c_mat.Write(queue, args.c_size, buffers_host.c_mat); }
    else if (name == kBufMatAP) { buffers.ap_mat.Write(queue, args.ap_size, buffers_host.ap_mat); }
    else if (name == kBufScalar) { buffers.scalar.Write(queue, args.scalar_size, buffers_host.scalar); }
    else { throw std::runtime_error("Invalid buffer name"); }
  }
}

// Compiles the above functions
template void DeviceToHost(const Arguments<half>&, Buffers<half>&, BuffersHost<half>&, Queue&, const std::vector<std::string>&);
template void DeviceToHost(const Arguments<float>&, Buffers<float>&, BuffersHost<float>&, Queue&, const std::vector<std::string>&);
template void DeviceToHost(const Arguments<double>&, Buffers<double>&, BuffersHost<double>&, Queue&, const std::vector<std::string>&);
template void DeviceToHost(const Arguments<float>&, Buffers<float2>&, BuffersHost<float2>&, Queue&, const std::vector<std::string>&);
template void DeviceToHost(const Arguments<double>&, Buffers<double2>&, BuffersHost<double2>&, Queue&, const std::vector<std::string>&);
template void DeviceToHost(const Arguments<float2>&, Buffers<float2>&, BuffersHost<float2>&, Queue&, const std::vector<std::string>&);
template void DeviceToHost(const Arguments<double2>&, Buffers<double2>&, BuffersHost<double2>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<half>&, Buffers<half>&, BuffersHost<half>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<float>&, Buffers<float>&, BuffersHost<float>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<double>&, Buffers<double>&, BuffersHost<double>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<float>&, Buffers<float2>&, BuffersHost<float2>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<double>&, Buffers<double2>&, BuffersHost<double2>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<float2>&, Buffers<float2>&, BuffersHost<float2>&, Queue&, const std::vector<std::string>&);
template void HostToDevice(const Arguments<double2>&, Buffers<double2>&, BuffersHost<double2>&, Queue&, const std::vector<std::string>&);

// =================================================================================================

// Conversion between half and single-precision
std::vector<float> HalfToFloatBuffer(const std::vector<half>& source) {
  auto result = std::vector<float>(source.size());
  for (auto i = size_t(0); i < source.size(); ++i) { result[i] = HalfToFloat(source[i]); }
  return result;
}
void FloatToHalfBuffer(std::vector<half>& result, const std::vector<float>& source) {
  for (auto i = size_t(0); i < source.size(); ++i) { result[i] = FloatToHalf(source[i]); }
}

// As above, but now for OpenCL data-types instead of std::vectors
Buffer<float> HalfToFloatBuffer(const Buffer<half>& source, cl_command_queue queue_raw) {
  const auto size = source.GetSize() / sizeof(half);
  auto queue = Queue(queue_raw);
  auto context = queue.GetContext();
  auto source_cpu = std::vector<half>(size);
  source.Read(queue, size, source_cpu);
  auto result_cpu = HalfToFloatBuffer(source_cpu);
  auto result = Buffer<float>(context, size);
  result.Write(queue, size, result_cpu);
  return result;
}
void FloatToHalfBuffer(Buffer<half>& result, const Buffer<float>& source, cl_command_queue queue_raw) {
  const auto size = source.GetSize() / sizeof(float);
  auto queue = Queue(queue_raw);
  auto context = queue.GetContext();
  auto source_cpu = std::vector<float>(size);
  source.Read(queue, size, source_cpu);
  auto result_cpu = std::vector<half>(size);
  FloatToHalfBuffer(result_cpu, source_cpu);
  result.Write(queue, size, result_cpu);
}

// Converts a 'real' value to a 'real argument' value to be passed to a kernel. Normally there is
// no conversion, but half-precision is not supported as kernel argument so it is converted to float.
template <> typename RealArg<half>::Type GetRealArg(const half value) { return HalfToFloat(value); }
template <> typename RealArg<float>::Type GetRealArg(const float value) { return value; }
template <> typename RealArg<double>::Type GetRealArg(const double value) { return value; }
template <> typename RealArg<float2>::Type GetRealArg(const float2 value) { return value; }
template <> typename RealArg<double2>::Type GetRealArg(const double2 value) { return value; }

// =================================================================================================

// Rounding functions performing ceiling and division operations
size_t CeilDiv(const size_t x, const size_t y) {
  return 1 + ((x - 1) / y);
}
size_t Ceil(const size_t x, const size_t y) {
  return CeilDiv(x,y)*y;
}

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(const size_t a, const size_t b) {
  return ((a/b)*b == a) ? true : false;
};

// =================================================================================================

// Convert the precision enum (as integer) into bytes
size_t GetBytes(const Precision precision) {
  switch(precision) {
    case Precision::kHalf: return 2;
    case Precision::kSingle: return 4;
    case Precision::kDouble: return 8;
    case Precision::kComplexSingle: return 8;
    case Precision::kComplexDouble: return 16;
    case Precision::kAny: return -1;
  }
}

// Convert the template argument into a precision value
template <> Precision PrecisionValue<half>() { return Precision::kHalf; }
template <> Precision PrecisionValue<float>() { return Precision::kSingle; }
template <> Precision PrecisionValue<double>() { return Precision::kDouble; }
template <> Precision PrecisionValue<float2>() { return Precision::kComplexSingle; }
template <> Precision PrecisionValue<double2>() { return Precision::kComplexDouble; }

// =================================================================================================

// Returns false is this precision is not supported by the device
template <> bool PrecisionSupported<float>(const Device &) { return true; }
template <> bool PrecisionSupported<float2>(const Device &) { return true; }
template <> bool PrecisionSupported<double>(const Device &device) {
  auto extensions = device.Capabilities();
  return (extensions.find(kKhronosDoublePrecision) == std::string::npos) ? false : true;
}
template <> bool PrecisionSupported<double2>(const Device &device) {
  auto extensions = device.Capabilities();
  return (extensions.find(kKhronosDoublePrecision) == std::string::npos) ? false : true;
}
template <> bool PrecisionSupported<half>(const Device &device) {
  auto extensions = device.Capabilities();
  if (device.Name() == "Mali-T628") { return true; } // supports fp16 but not cl_khr_fp16 officially
  return (extensions.find(kKhronosHalfPrecision) == std::string::npos) ? false : true;
}

// =================================================================================================
} // namespace clblast

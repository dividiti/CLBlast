#ifndef DVDT_INFER_H
#define DVDT_INFER_H

#include <vector>

#include "clblast.h"

namespace clblast{


	 template <typename T> 
    const std::vector<std::string> GetConf(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const T alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const T beta, const size_t c_offset, const size_t c_ld){

      std::vector<std::string> routines_vett = {"Copy","Pad","Transpose",
                      "Padtranspose","KernelSelection"};

      routines_vett.push_back("XgemmDirect");
      routines_vett.push_back("Xgemm"); 
      return routines_vett;
    }

    template const std::vector<std::string> GetConf<float>(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const float alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const float beta, const size_t c_offset, const size_t c_ld);

    template const std::vector<std::string> GetConf<double>(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const double alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const double beta, const size_t c_offset, const size_t c_ld);

    template const std::vector<std::string> GetConf<float2>(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const float2 alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const float2 beta, const size_t c_offset, const size_t c_ld);

    template const std::vector<std::string> GetConf<double2>(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const double2 alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const double2 beta, const size_t c_offset, const size_t c_l);

    template const std::vector<std::string> GetConf<half>(const Layout layout, const Transpose a_transpose, 
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const half alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const half beta, const size_t c_offset, const size_t c_ld); 
}
#endif
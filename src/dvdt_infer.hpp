#ifndef DVDT_INFER_H
#define DVDT_INFER_H

#include <vector>
#include <initializer_list>
#include "clblast.h"

namespace clblast{
    
    struct dvdtKernelInfo{
        std::vector<std::string> routines_vett;
        std::initializer_list<const char *> sources;
    };

    std::initializer_list<const char *> getSources( const char * elem) {
        std::initializer_list<const char *> sources = {
        "#include \"../../kernels/level3/level3.opencl\""
        "#include \"../../kernels/level3/copy_fast.opencl\""
        "#include \"../../kernels/level3/copy_pad.opencl\""
        "#include \"../../kernels/level3/transpose_fast.opencl\""
        "#include \"../../kernels/level3/transpose_pad.opencl\""
        "#include \"../../kernels/level3/convert_symmetric.opencl\""
        "#include \"../../kernels/level3/convert_triangular.opencl\""
        "#include \"../../kernels/level3/convert_hermitian.opencl\""
        , // separated in multiple parts to prevent C1091 in MSVC 2013
        "#include \"../../kernels/level3/xgemm_direct_part1.opencl\""
        "#include \"../../kernels/level3/xgemm_direct_part2.opencl\""
        "#include \"../../kernels/level3/xgemm_direct_part3.opencl\""
        , // separated in multiple parts to prevent C1091 in MSVC 2013
        "#include \"../../kernels/level3/xgemm_part1.opencl\""
        "#include \"../../kernels/level3/xgemm_part2.opencl\"",elem
        };
        
        return sources;
    }

	template <typename T> 
    struct dvdtKernelInfo GetConf(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const T alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const T beta, const size_t c_offset, const size_t c_ld, int * flag){

      std::vector<std::string> routines_vett = {"Copy","Pad","Transpose",
                      "Padtranspose","KernelSelection"};

      routines_vett.push_back("XgemmDirect");
      routines_vett.push_back("Xgemm");
      *flag=-1;
      struct dvdtKernelInfo k_info;
      k_info.routines_vett = routines_vett;
      k_info.sources = getSources("#include \"../../kernels/level3/xgemm_part3.opencl\"");
      return k_info;
    }

    template struct dvdtKernelInfo GetConf<float>(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const float alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const float beta, const size_t c_offset, const size_t c_ld, int * flag);

    template struct dvdtKernelInfo GetConf<double>(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const double alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const double beta, const size_t c_offset, const size_t c_ld, int * flag);

    template struct dvdtKernelInfo GetConf<float2>(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const float2 alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const float2 beta, const size_t c_offset, const size_t c_ld, int * flag);

    template struct dvdtKernelInfo GetConf<double2>(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const double2 alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const double2 beta, const size_t c_offset, const size_t c_l, int * flag);

    template struct dvdtKernelInfo GetConf<half>(const Layout layout, const Transpose a_transpose, 
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const half alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const half beta, const size_t c_offset, const size_t c_ld, int * flag); 

  
}

#endif




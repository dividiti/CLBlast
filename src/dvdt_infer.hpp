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
                    const T beta, const size_t c_offset, const size_t c_ld, int * flag){

      std::vector<std::string> routines_vett = {"Copy","Pad","Transpose",
                      "Padtranspose","KernelSelection"};

      routines_vett.push_back("XgemmDirect");
      routines_vett.push_back("Xgemm"); 
      *flag=-1;
      return routines_vett;
    }

    template const std::vector<std::string> GetConf<float>(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const float alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const float beta, const size_t c_offset, const size_t c_ld, int * flag);

    template const std::vector<std::string> GetConf<double>(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const double alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const double beta, const size_t c_offset, const size_t c_ld, int * flag);

    template const std::vector<std::string> GetConf<float2>(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const float2 alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const float2 beta, const size_t c_offset, const size_t c_ld, int * flag);

    template const std::vector<std::string> GetConf<double2>(const Layout layout, const Transpose a_transpose, 
                    const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                    const double2 alpha, const size_t a_offset, const size_t a_ld,
                    const size_t b_offset, const size_t b_ld,
                    const double2 beta, const size_t c_offset, const size_t c_l, int * flag);

    template const std::vector<std::string> GetConf<half>(const Layout layout, const Transpose a_transpose, 
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const half alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const half beta, const size_t c_offset, const size_t c_ld, int * flag); 


    template <typename T> void testConf<T>(const Layout layout, const Transpose a_transpose, 
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const half alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const half beta, const size_t c_offset, const size_t c_ld, const len, 
                const std::vector<std::string> v){
        
        auto platform = Platform(size{0});
        auto device = Device(platform, size{0});
        auto context = Context(device);
        auto queue = Queue(context, device);
        for(auto i = 0; i < len ; i++)
        {
            std::vector<std::string> routines_vett = {"Copy","Pad","Transpose",
                      "Padtranspose","KernelSelection"};
            routines_vett.push_back(v[i]);
            try {
                auto queue_plain = queue();
                auto event = cl_event{};
                auto queue_cpp = Queue(*queue);
                int flag = -1;
                const std::vector<std::string> routines_vett = 
                GetConf<T>(layout, a_transpose, b_transpose,
                               m, n,k, 
                               alpha, a_offset, a_ld,
                               b_offset, b_ld, beta,
                               c_offset, c_ld, &flag);

                auto routine = Xgemm<T>(queue_cpp, event, routines_vett);
                
                routine.DoGemm(layout, a_transpose, b_transpose,
                               m, n, k,
                               alpha,
                               Buffer<T>(a_buffer), a_offset, a_ld,
                               Buffer<T>(b_buffer), b_offset, b_ld,
                               beta,
                               Buffer<T>(c_buffer), c_offset, c_ld);
                return StatusCode::kSuccess;
              } catch (...) { return DispatchException(); }
        
        }

        template void testConf<float>(const Layout layout, const Transpose a_transpose, 
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const float alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const float beta, const size_t c_offset, const size_t c_ld, const len, 
                const std::vector<std::string> v);

         template void testConf<double>(const Layout layout, const Transpose a_transpose, 
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const double alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const double beta, const size_t c_offset, const size_t c_ld, const len, 
                const std::vector<std::string> v);

          template void testConf<float2>(const Layout layout, const Transpose a_transpose, 
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const float2 alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const float2 beta, const size_t c_offset, const size_t c_ld, const len, 
                const std::vector<std::string> v);

           template void testConf<double2>(const Layout layout, const Transpose a_transpose, 
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const double2 alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const double2 beta, const size_t c_offset, const size_t c_ld, const len, 
                const std::vector<std::string> v);

            template void testConf<half>(const Layout layout, const Transpose a_transpose, 
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const half alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const half beta, const size_t c_offset, const size_t c_ld, const len, 
                const std::vector<std::string> v);
}

#endif





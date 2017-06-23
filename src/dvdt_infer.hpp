#ifndef DVDT_INFER_H
#define DVDT_INFER_H

template <typename T> 
const std::vector<std::string> GetConf(const Layout layout, const Transpose a_transpose, 
                const Transpose b_transpose, const size_t m, const size_t n, const size_t k,
                const T alpha, const size_t a_offset, const size_t a_ld,
                const size_t b_offset, const size_t b_ld,
                const T beta, const size_t c_offset, const size_t c_ld);
#endif
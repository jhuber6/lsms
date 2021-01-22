#ifndef ULMBLAS_CXXBLAS_LEVEL3_SYMM_H
#define ULMBLAS_CXXBLAS_LEVEL3_SYMM_H 1

namespace cxxblas {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
    void
    symm(bool         left,
         IndexType    m,
         IndexType    n,
         const Alpha  &alpha,
         bool         lowerA,
         const TA     *A,
         IndexType    incRowA,
         IndexType    incColA,
         const TB     *B,
         IndexType    incRowB,
         IndexType    incColB,
         const Beta   &beta,
         TC           *C,
         IndexType    incRowC,
         IndexType    incColC);

} // namespace cxxblas

#endif // ULMBLAS_CXXBLAS_LEVEL3_SYMM_H

#include <ulmblas/cxxblas/level3/symm.tcc>

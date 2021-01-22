#ifndef ULMBLAS_CXXBLAS_LEVEL1_ROT_TCC
#define ULMBLAS_CXXBLAS_LEVEL1_ROT_TCC 1

#include <ulmblas/cxxblas/level1/rot.h>
#include <ulmblas/impl/level1/rot.h>

namespace cxxblas {

template <typename A, typename B, typename T>
void
rotg(A &a,
     B &b,
     T &c,
     T &s)
{
    ulmBLAS::rotg(a, b, c, s);
}

template <typename TA, typename TB, typename T>
void
rotg(std::complex<TA>   &a,
     std::complex<TB>   &b,
     T                  &c,
     std::complex<T>    &s)
{
    ulmBLAS::rotg(a, b, c, s);
}

template <typename IndexType, typename VX, typename VY, typename T>
void
rot(IndexType   n,
    VX          *x,
    IndexType   incX,
    VY          *y,
    IndexType   incY,
    T           c,
    T           s)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rot(n, x, incX, y, incY, c, s);
}

template <typename IndexType, typename X, typename Y, typename T>
void
rot(IndexType              n,
    std::complex<X>        *x,
    IndexType              incX,
    std::complex<Y>        *y,
    IndexType              incY,
    T                      c,
    const std::complex<T>  &s)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (incY<0) {
        y -= incY*(n-1);
    }
    ulmBLAS::rot(n, x, incX, y, incY, c, s);
}

} // namespace cxxblas

#endif // ULMBLAS_CXXBLAS_LEVEL1_ROT_TCC

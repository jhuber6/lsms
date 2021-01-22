/*
 * Copyright (C) 2014, The University of Texas at Austin
 * Copyright (C) 2014-2015, Michael Lehn
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas at Austin nor the names
 *    of its contributors may be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef ULMBLAS_IMPL_LEVEL1_KERNEL_REF_AXPY_TCC
#define ULMBLAS_IMPL_LEVEL1_KERNEL_REF_AXPY_TCC 1

#include <complex>
#include <ulmblas/impl/auxiliary/conjugate.h>
#include <ulmblas/impl/level1/kernel/ref/axpy.h>

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename Alpha, typename VX, typename VY>
void
axpy(IndexType      n,
     const Alpha    &alpha,
     const VX       *x,
     IndexType      incX,
     VY             *y,
     IndexType      incY)
{
    if (n<=0 || alpha==Alpha(0)) {
        return;
    }

    for (IndexType i=0; i<n; ++i) {
        y[i*incY] += alpha*x[i*incX];
    }
}

template <typename IndexType, typename Alpha, typename VX, typename VY>
void
axpy(IndexType                  n,
     const std::complex<Alpha>  &alpha,
     const std::complex<VX>     *x,
     IndexType                  incX,
     std::complex<VY>           *y,
     IndexType                  incY)
{
    if (n<=0 || alpha==Alpha(0)) {
        return;
    }

    Alpha     ra = alpha.real();
    Alpha     ia = alpha.imag();

    const VX *rx = reinterpret_cast<const VX *>(x);
    const VX *ix = rx + 1;

    VY       *ry = reinterpret_cast<VY *>(y);
    VY       *iy = ry + 1;

    incX *= 2;
    incY *= 2;

    for (IndexType i=0; i<n; ++i) {
        ry[i*incY] += ra*rx[i*incX] - ia*ix[i*incX];
        iy[i*incY] += ra*ix[i*incX] + ia*rx[i*incX];
    }
}

template <typename IndexType, typename Alpha, typename VX, typename VY>
void
acxpy(IndexType      n,
      const Alpha    &alpha,
      const VX       *x,
      IndexType      incX,
      VY             *y,
      IndexType      incY)
{
    if (n<=0 || alpha==Alpha(0)) {
        return;
    }

    for (IndexType i=0; i<n; ++i) {
        y[i*incY] += alpha*conjugate(x[i*incX]);
    }
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_IMPL_LEVEL1_KERNEL_REF_AXPY_TCC 1

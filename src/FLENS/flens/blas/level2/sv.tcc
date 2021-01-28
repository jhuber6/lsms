/*
 *   Copyright (c) 2010, Michael Lehn
 *
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1) Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2) Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *   3) Neither the name of the FLENS development group nor the names of
 *      its contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef FLENS_BLAS_LEVEL2_SV_TCC
#define FLENS_BLAS_LEVEL2_SV_TCC

#include <flens/blas/closures/closures.h>
#include <flens/blas/level2/level2.h>
#include <flens/typedefs.h>
#include <ulmblas/cxxblas.h>

#ifdef FLENS_DEBUG_CLOSURES
#   include <flens/blas/blaslogon.h>
#else
#   include <flens/blas/blaslogoff.h>
#endif


namespace flens { namespace blas {

//-- tbsv
template <typename MA, typename VX>
typename RestrictTo<IsTbMatrix<MA>::value
                 && IsDenseVector<VX>::value,
         void>::Type
sv(Transpose transposeA, const MA &A, VX &&x)
{
    ASSERT(x.length()==A.dim());

    const bool colMajorA = (A.order()==ColMajor);
    const bool lowerA    = (A.upLo()==Lower);
    const bool transA    = (transposeA==Trans || transposeA==ConjTrans);
    const bool conjA     = (transposeA==Conj || transposeA==ConjTrans);
    const bool unitDiagA = (A.diag()==Unit);

    cxxblas::tbsv(A.dim(), A.numOffDiags(),
                  colMajorA, lowerA, transA, conjA, unitDiagA,
                  A.data(), A.leadingDimension(),
                  x.data(), x.stride());
}

//-- trsv
template <typename MA, typename VX>
typename RestrictTo<IsTrMatrix<MA>::value
                 && IsDenseVector<VX>::value,
         void>::Type
sv(Transpose transposeA, const MA &A, VX &&x)
{
    const bool lowerA    = (A.upLo()==Lower);
    const bool transA    = (transposeA==Trans || transposeA==ConjTrans);
    const bool conjA     = (transposeA==Conj || transposeA==ConjTrans);
    const bool unitDiagA = (A.diag()==Unit);

    ASSERT(x.length()==A.dim());

    cxxblas::trsv(A.dim(),
                  lowerA, transA, conjA, unitDiagA,
                  A.data(), A.strideRow(), A.strideCol(),
                  x.data(), x.stride());
}

//-- tpsv
template <typename MA, typename VX>
typename RestrictTo<IsTpMatrix<MA>::value
                 && IsDenseVector<VX>::value,
         void>::Type
sv(Transpose transposeA, const MA &A, VX &&x)
{
    const bool colMajorA = (A.order()==ColMajor);
    const bool lowerA    = (A.upLo()==Lower);
    const bool transA    = (transposeA==Trans || transposeA==ConjTrans);
    const bool conjA     = (transposeA==Conj || transposeA==ConjTrans);
    const bool unitDiagA = (A.diag()==Unit);

    ASSERT(x.length()==A.dim());

    cxxblas::tpsv(A.dim(),
                  colMajorA, lowerA, transA, conjA, unitDiagA,
                  A.data(),
                  x.data(), x.stride());
}

} } // namespace blas, flens

#endif // FLENS_BLAS_LEVEL3_SV_TCC
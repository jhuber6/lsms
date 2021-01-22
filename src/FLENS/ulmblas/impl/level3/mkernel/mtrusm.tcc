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

#ifndef ULMBLAS_IMPL_LEVEL3_MKERNEL_MTRUSM_TCC
#define ULMBLAS_IMPL_LEVEL3_MKERNEL_MTRUSM_TCC 1

#include <ulmblas/impl/level3/mkernel/mtrusm.h>
#include <ulmblas/impl/level3/ukernel/ugemm.h>
#include <ulmblas/impl/level3/ukernel/utrusm.h>

namespace ulmBLAS {

template <typename IndexType, typename T, typename TB>
void
mtrusm(IndexType    mc,
       IndexType    nc,
       const T      &alpha,
       const T      *A_,
       T            *B_,
       TB           *B,
       IndexType    incRowB,
       IndexType    incColB)
{
    const IndexType MR = BlockSize<T>::MR;
    const IndexType NR = BlockSize<T>::NR;

    const IndexType mp = (mc+MR-1) / MR;
    const IndexType np = (nc+NR-1) / NR;

    const IndexType mr_ = mc % MR;
    const IndexType nr_ = nc % NR;

    IndexType mr, nr;
    IndexType kc;

    const T *nextA = nullptr;
    const T *nextB = nullptr;

    const IndexType na = mp*(2*mc-(mp-1)*MR)/2;

#pragma omp target teams distribute parallel for map(to:alpha)
    for (IndexType j=0; j<np; ++j) {
        IndexType ia = na;
        IndexType ib = mc;

        for (IndexType i=mp-1; i>=0; --i) {
            nr    = (j!=np-1 || nr_==0) ? NR : nr_;
            mr    = (i!=mp-1 || mr_==0) ? MR : mr_;
            kc    = std::max(mc-(i+1)*MR, IndexType(0));

            ia    -= mr + kc;
            ib    -= mr;

            if (mr==MR && nr==NR) {
                ugemm(kc,
                      T(-1), &A_[(ia+MR)*MR], &B_[(j*mc+ib+MR)*NR],
                      alpha,
                      &B_[(j*mc+ib)*NR], NR, IndexType(1),
                      nextA, nextB);

                utrusm(&A_[ia*MR], &B_[(j*mc+ib)*NR],
                       &B_[(j*mc+ib)*NR], NR, IndexType(1));
            } else {

                // Call buffered micro kernels

                ugemm(mr, nr, kc,
                      T(-1), &A_[(ia+MR)*MR], &B_[(j*mc+ib+mr)*NR],
                      alpha,
                      &B_[(j*mc+ib)*NR], NR, IndexType(1),
                      nextA, nextB);

                utrusm(mr, nr,
                       &A_[ia*MR], &B_[(j*mc+ib)*NR],
                       &B_[(j*mc+ib)*NR], NR, IndexType(1));
            }
        }
    }
    for (IndexType j=0; j<np; ++j) {
        nr    = (j!=np-1 || nr_==0) ? NR : nr_;

        gecopy(mc, nr,
               &B_[j*mc*NR], NR, IndexType(1),
               &B[j*NR*incColB], incRowB, incColB);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_IMPL_LEVEL3_MKERNEL_MTRUSM_TCC

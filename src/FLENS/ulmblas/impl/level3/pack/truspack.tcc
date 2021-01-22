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

#ifndef ULMBLAS_IMPL_LEVEL3_PACK_TRUSPACK_TCC
#define ULMBLAS_IMPL_LEVEL3_PACK_TRUSPACK_TCC 1

#include <ulmblas/impl/auxiliary/conjugate.h>
#include <ulmblas/impl/level3/pack/trlpack.h>
#include <ulmblas/impl/level3/ukernel/ugemm.h>

namespace ulmBLAS {

template <typename IndexType, typename TU, typename Buffer>
void
truspack(IndexType   mc,
         bool        conj,
         bool        unit,
         const TU    *U,
         IndexType   incRowU,
         IndexType   incColU,
         Buffer      *p)
{
    const IndexType MR  = BlockSize<Buffer>::MR;
    const IndexType mp  = mc / MR;

#pragma omp target teams distribute collapse(2)
    for (IndexType i=0; i<mp; ++i) {
        for (IndexType i0=0; i0<MR; ++i0) {
#pragma omp parallel for collapse(2)
            for (IndexType j=i; j<mp; ++j) {
                for (IndexType j0=0; j0<MR; ++j0) {
                    IndexType I  = i*MR+i0;
                    IndexType J  = j*MR+j0;
                    IndexType nu = ((mp+1)*mp/2-(mp-i+1)*(mp-i)/2)*MR*MR + (j-i)*MR*MR + j0*MR +i0;
                    p[nu] = (I==J && unit)
                            ? Buffer(1)
                          : (I==J && !unit)
                            ? Buffer(1) / conjugate(U[I*(incRowU+incColU)], conj)
                          : (I>=mc || J>=mc)
                            ? Buffer(0)
                          : (I<J)
                            ? conjugate(U[I*incRowU+J*incColU], conj)
                          : Buffer(0);
                }
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_IMPL_LEVEL3_PACK_TRUSPACK_TCC

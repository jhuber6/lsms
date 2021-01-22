#include <flens/blas/interface/blas/config.h>


using namespace flens;

extern "C" {

void
BLAS(sgemv)(const char      *TRANS,
            const INTEGER   *M,
            const INTEGER   *N,
            const float     *ALPHA,
            const float     *A_,
            const INTEGER   *LDA,
            const float     *X,
            const INTEGER   *INCX,
            const float     *BETA,
            float           *Y,
            const INTEGER   *INCY)
{
    BLAS_DEBUG_OUT("BLAS INTERFACE: sgemv");

#   ifdef TEST_DIRECT_CBLAS

        const char         TRANS_ = toupper(*TRANS);
        const Transpose    trans  = convertTo<Transpose>(TRANS_);

        cblas_sgemv(CBLAS_ORDER::CblasColMajor,
                    cxxblas::CBLAS::getCblasType(trans),
                    *M, *N, *ALPHA, A_, *LDA, X, *INCX, *BETA, Y, *INCY);

#   else

        using std::abs;
        using std::max;

        char    TRANS_ = toupper(*TRANS);
#       ifndef NO_INPUT_CHECK
            INTEGER info    = 0;
            if (TRANS_!='N' && TRANS_!='T' && TRANS_!='C') {
                info = 1;
            } else if (*M<0) {
                info = 2;
            } else if (*N<0) {
                info = 3;
            } else if (*LDA<max(INTEGER(1),*M)) {
                info = 6;
            } else if (*INCX==0) {
                info = 8;
            } else if (*INCY==0) {
                info = 11;
            }
            if (info!=0) {
                BLAS(xerbla)("SGEMV ", &info);
                return;
            }
#       endif

        Transpose    trans = convertTo<Transpose>(TRANS_);
        const bool   noTrans = (trans==NoTrans || trans==Conj);
        INTEGER      lenX, lenY;

        if (noTrans) {
            lenX = *N;
            lenY = *M;
        } else {
            lenX = *M;
            lenY = *N;
        }

        SGeMatrixConstView    A = SFullConstView(*M, *N, *LDA, A_);
        SDenseVectorConstView x(SConstArrayView(lenX, X, abs(*INCX)), *INCX<0);
        SDenseVectorView      y(SArrayView(lenY, Y, abs(*INCY)), *INCY<0);

#       ifdef TEST_OVERLOADED_OPERATORS
            const auto alpha = *ALPHA;
            const auto beta  = *BETA;

            if (trans==NoTrans) {
                y = beta*y + alpha*A*x;
            } else if (trans==Trans) {
                y = beta*y + alpha*transpose(A)*x;
            } else if (trans==ConjTrans) {
                y = beta*y + alpha*conjTrans(A)*x;
            }
#       else
            blas::mv(trans, *ALPHA, A, x, *BETA, y);
#       endif
#   endif
}

void
BLAS(dgemv)(const char      *TRANS,
            const INTEGER   *M,
            const INTEGER   *N,
            const double    *ALPHA,
            const double    *A_,
            const INTEGER   *LDA,
            const double    *X,
            const INTEGER   *INCX,
            const double    *BETA,
            double          *Y,
            const INTEGER   *INCY)
{
    BLAS_DEBUG_OUT("BLAS INTERFACE: dgemv");

#   ifdef TEST_DIRECT_CBLAS

        const char         TRANS_ = toupper(*TRANS);
        const Transpose    trans  = convertTo<Transpose>(TRANS_);

        cblas_dgemv(CBLAS_ORDER::CblasColMajor,
                    cxxblas::CBLAS::getCblasType(trans),
                    *M, *N, *ALPHA, A_, *LDA, X, *INCX, *BETA, Y, *INCY);

#   else

        using std::abs;
        using std::max;

        INTEGER info   = 0;
        char    TRANS_ = toupper(*TRANS);

        if (TRANS_!='N' && TRANS_!='T' && TRANS_!='C') {
            info = 1;
        } else if (*M<0) {
            info = 2;
        } else if (*N<0) {
            info = 3;
        } else if (*LDA<max(INTEGER(1),*M)) {
            info = 6;
        } else if (*INCX==0) {
            info = 8;
        } else if (*INCY==0) {
            info = 11;
        }
        if (info!=0) {
            BLAS(xerbla)("DGEMV ", &info);
            return;
        }

        Transpose    trans = convertTo<Transpose>(TRANS_);
        const bool   noTrans = (trans==NoTrans || trans==Conj);
        INTEGER      lenX, lenY;

        if (noTrans) {
            lenX = *N;
            lenY = *M;
        } else {
            lenX = *M;
            lenY = *N;
        }

        DGeMatrixConstView    A = DFullConstView(*M, *N, *LDA, A_);
        DDenseVectorConstView x(DConstArrayView(lenX, X, abs(*INCX)), *INCX<0);
        DDenseVectorView      y(DArrayView(lenY, Y, abs(*INCY)), *INCY<0);

#       ifdef TEST_OVERLOADED_OPERATORS
            const auto alpha = *ALPHA;
            const auto beta  = *BETA;

            if (trans==NoTrans) {
                y = beta*y + alpha*A*x;
            } else if (trans==Trans) {
                y = beta*y + alpha*transpose(A)*x;
            } else if (trans==ConjTrans) {
                y = beta*y + alpha*conjTrans(A)*x;
            }
#       else
            blas::mv(trans, *ALPHA, A, x, *BETA, y);
#       endif
#   endif
}

void
BLAS(cgemv)(const char      *TRANS,
            const INTEGER   *M,
            const INTEGER   *N,
            const cfloat    *ALPHA,
            const cfloat    *A_,
            const INTEGER   *LDA,
            const cfloat    *X,
            const INTEGER   *INCX,
            const cfloat    *BETA,
            cfloat          *Y,
            const INTEGER   *INCY)
{
    BLAS_DEBUG_OUT("BLAS INTERFACE: cgemv");

#   ifdef TEST_DIRECT_CBLAS

        const char         TRANS_ = toupper(*TRANS);
        const Transpose    trans  = convertTo<Transpose>(TRANS_);

        cblas_cgemv(CBLAS_ORDER::CblasColMajor,
                    cxxblas::CBLAS::getCblasType(trans),
                    *M, *N,
                    reinterpret_cast<const float *>(ALPHA),
                    reinterpret_cast<const float *>(A_), *LDA,
                    reinterpret_cast<const float*>(X), *INCX,
                    reinterpret_cast<const float*>(BETA),
                    reinterpret_cast<float*>(Y), *INCY);

#   else

        using std::abs;
        using std::max;

        INTEGER info   = 0;
        char    TRANS_ = toupper(*TRANS);

        if (TRANS_!='N' && TRANS_!='T' && TRANS_!='C') {
            info = 1;
        } else if (*M<0) {
            info = 2;
        } else if (*N<0) {
            info = 3;
        } else if (*LDA<max(INTEGER(1),*M)) {
            info = 6;
        } else if (*INCX==0) {
            info = 8;
        } else if (*INCY==0) {
            info = 11;
        }
        if (info!=0) {
            BLAS(xerbla)("CGEMV ", &info);
            return;
        }

        Transpose    trans = convertTo<Transpose>(TRANS_);
        const bool   noTrans = (trans==NoTrans || trans==Conj);
        INTEGER      lenX, lenY;

        if (noTrans) {
            lenX = *N;
            lenY = *M;
        } else {
            lenX = *M;
            lenY = *N;
        }

        CGeMatrixConstView    A = CFullConstView(*M, *N, *LDA, A_);
        CDenseVectorConstView x(CConstArrayView(lenX, X, abs(*INCX)), *INCX<0);
        CDenseVectorView      y(CArrayView(lenY, Y, abs(*INCY)), *INCY<0);

#       ifdef TEST_OVERLOADED_OPERATORS
            const auto alpha = *ALPHA;
            const auto beta  = *BETA;

            if (trans==NoTrans) {
                y = beta*y + alpha*A*x;
            } else if (trans==Trans) {
                y = beta*y + alpha*transpose(A)*x;
            } else if (trans==ConjTrans) {
                y = beta*y + alpha*conjTrans(A)*x;
            }
#       else
            blas::mv(trans, *ALPHA, A, x, *BETA, y);
#       endif
#   endif
}

void
BLAS(zgemv)(const char      *TRANS,
            const INTEGER   *M,
            const INTEGER   *N,
            const cdouble   *ALPHA,
            const cdouble   *A_,
            const INTEGER   *LDA,
            const cdouble   *X,
            const INTEGER   *INCX,
            const cdouble   *BETA,
            cdouble         *Y,
            const INTEGER   *INCY)
{
    BLAS_DEBUG_OUT("BLAS INTERFACE: zgemv");

#   ifdef TEST_DIRECT_CBLAS

        const char         TRANS_ = toupper(*TRANS);
        const Transpose    trans  = convertTo<Transpose>(TRANS_);

        cblas_zgemv(CBLAS_ORDER::CblasColMajor,
                    cxxblas::CBLAS::getCblasType(trans),
                    *M, *N,
                    reinterpret_cast<const double *>(ALPHA),
                    reinterpret_cast<const double *>(A_), *LDA,
                    reinterpret_cast<const double*>(X), *INCX,
                    reinterpret_cast<const double*>(BETA),
                    reinterpret_cast<double*>(Y), *INCY);

#   else

        using std::abs;
        using std::max;

        INTEGER info   = 0;
        char    TRANS_ = toupper(*TRANS);

        if (TRANS_!='N' && TRANS_!='T' && TRANS_!='C') {
            info = 1;
        } else if (*M<0) {
            info = 2;
        } else if (*N<0) {
            info = 3;
        } else if (*LDA<max(INTEGER(1),*M)) {
            info = 6;
        } else if (*INCX==0) {
            info = 8;
        } else if (*INCY==0) {
            info = 11;
        }
        if (info!=0) {
            BLAS(xerbla)("ZGEMV ", &info);
            return;
        }

        Transpose    trans = convertTo<Transpose>(TRANS_);
        const bool   noTrans = (trans==NoTrans || trans==Conj);
        INTEGER      lenX, lenY;

        if (noTrans) {
            lenX = *N;
            lenY = *M;
        } else {
            lenX = *M;
            lenY = *N;
        }

        ZGeMatrixConstView    A = ZFullConstView(*M, *N, *LDA, A_);
        ZDenseVectorConstView x(ZConstArrayView(lenX, X, abs(*INCX)), *INCX<0);
        ZDenseVectorView      y(ZArrayView(lenY, Y, abs(*INCY)), *INCY<0);

#       ifdef TEST_OVERLOADED_OPERATORS
            const auto alpha = *ALPHA;
            const auto beta  = *BETA;

            if (trans==NoTrans) {
                y = beta*y + alpha*A*x;
            } else if (trans==Trans) {
                y = beta*y + alpha*transpose(A)*x;
            } else if (trans==ConjTrans) {
                y = beta*y + alpha*conjTrans(A)*x;
            }
#       else
            blas::mv(trans, *ALPHA, A, x, *BETA, y);
#       endif
#   endif
}


} // extern "C"

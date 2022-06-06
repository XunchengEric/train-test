#ifndef _SCALAPACK_HPP_
#define _SCALAPACK_HPP_

#include "environment.h"

/// @namespace dgdft
///
/// @brief Main namespace for DGDFT routines.
namespace dgdft{

/// @namespace scalapack
///
/// @brief Thin interface to ScaLAPACK.
namespace scalapack{

typedef  int                    Int;
typedef  std::complex<float>    scomplex;
typedef  std::complex<double>   dcomplex;


extern "C"{

// *********************************************************************
// BLACS  routines
// *********************************************************************
void Cblacs_get(const Int contxt, const Int what, Int* val);

void Cblacs_gridinit(Int* contxt, const char* order, const Int nprow, const Int npcol);

void Cblacs_gridmap(Int* contxt, Int* pmap, const Int ldpmap, const Int nprow, const Int npcol);

void Cblacs_gridinfo(const Int contxt,  Int* nprow, Int* npcol, 
    Int* myprow, Int* mypcol);

void Cblacs_gridexit    (    int contxt );    


// *********************************************************************
// ScaLAPACK and PBLAS routines
// *********************************************************************

int SCALAPACK(numroc)(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);

void SCALAPACK(descinit)(Int* desc, const Int* m, const Int * n, const Int* mb,
    const Int* nb, const Int* irsrc, const Int* icsrc,
    const Int* contxt, const Int* lld, Int* info);


// Some Level-1 PBLAS routines
void SCALAPACK(pdnrm2)(const Int *n , double *norm2 , const double *x , const Int *ix , const Int *jx , const Int *descx , const Int *incx );
void SCALAPACK(pdscal)(const Int *n , const double *a , double *x , const Int *ix , const Int *jx , const Int *descx , const Int *incx );
void SCALAPACK(pddot)(const Int *n , double *dot , const double *x , const Int *ix , const Int *jx , const Int *descx , const Int *incx , 
		      const double *y , const Int *iy , const Int *jy , const Int *descy , const Int *incy );
void SCALAPACK(pdaxpy)(const Int *n , const double *a , const double *x , const Int *ix , const Int *jx , const Int *descx , const Int *incx , 
		       double *y , const Int *iy , const int *jy , const Int *descy , const Int *incy );
void SCALAPACK(pdcopy)(const Int *n , const double *x , const Int *ix , const Int *jx , const Int *descx , const Int *incx , 
		       double *y , const Int *iy , const Int *jy , const Int *descy , const Int *incy );


// Some Level-2 PBLAS routines
void SCALAPACK(pdgemv)(const char *trans , const Int *m , const Int *n , const double *alpha , const double *a , const Int *ia , const Int *ja , const Int *desca , 
		       const double *x , const Int *ix , const Int *jx , const Int *descx , const Int *incx , 
		       const double *beta , double *y , const Int *iy , const Int *jy , const Int *descy , const Int *incy );

void SCALAPACK(pdtrsv)(const char* uplo, const char* trans, const char* diag, const Int* m, const double* a, const Int* ia, const Int* ja, const Int* desca, 
                const double* x, const Int* ib, const Int* jb, const Int* descx, const Int* incx);

// Some Level-3 PBLAS routines
void SCALAPACK(pdgemm)(const char* transA, const char* transB,
    const Int* m, const Int* n, const Int* k,
    const double* alpha,
    const double* A, const Int* ia, const Int* ja, const Int* desca, 
    const double* B, const Int* ib, const Int* jb, const Int* descb,
    const double* beta,
    double* C, const Int* ic, const Int* jc, const Int* descc);

void SCALAPACK(pdtrmm)(const char* side, const char* uplo, const char* trans, const char* diag,
    const Int* m, const Int* n, const double* alpha,
    const double* A, const Int* ia, const Int* ja, const Int* desca, 
    const double* B, const Int* ib, const Int* jb, const Int* descb);

void SCALAPACK(pdsymm)(const char* side, const char* uplo,
    const Int* m, const Int* n, 
    const double* alpha,
    const double* A, const Int* ia, const Int* ja, const Int* desca, 
    const double* B, const Int* ib, const Int* jb, const Int* descb,
    const double* beta,
    const double* C, const Int* ic, const Int* jc, const Int* descc);

void SCALAPACK(pdgels)(char* transA,
    const Int* m, const Int* n, const Int* NRHS,
    const double* A, const Int* ia, const Int* ja, const Int* desca, 
    const double* B, const Int* ib, const Int* jb, const Int* descb,
    double *work, Int *lwork, Int *info);

void  SCALAPACK(pdsyrk)(const char *uplo , const char *trans , const Int *n , const Int *k , 
			const double *alpha , 
			const double *a , const Int *ia , const Int *ja , const Int *desca , 
			const double *beta , 
			double *c , const Int *ic , const Int *jc , const Int *descc);

void SCALAPACK(pdsyr2k)(const char *uplo , const char *trans , const Int *n , const Int *k , 
			const double *alpha , 
			const double *a , const Int *ia , const Int *ja , const Int *desca , 
			const double *b , const Int *ib , const Int *jb , const Int *descb , 
			const double *beta , 
			double *c , const Int *ic , const Int *jc , const Int *descc );

void SCALAPACK(pdtradd)(const char* uplo, const char* trans, const Int* m, const Int* n,
                        const double* alpha,
                        const double* a, const Int* ia, const Int* ja, const Int* desca, 
                        const double* beta,
                        double* c, const Int* ic, const Int* jc, const Int* descc);

void SCALAPACK(pdgeadd)(const char *trans,
    const Int* m, const Int* n, 
    const double *alpha, 
    const double* A, const Int* ia, const Int* ja, const Int* desca, 
    const double* beta,
    double* B, const Int* ib, const Int* jb, const Int* descb);

void SCALAPACK(pdtrsm)( const char* side, const char* uplo, 
    const char* trans, const char* diag,
    const int* m, const int* n, const double* alpha,
    const double* a, const int* ia, const int* ja, const int* desca, 
    double* b, const int* ib, const int* jb, const int* descb );


// Other ScaLAPACK routines

// Redistribution
void SCALAPACK(pdgemr2d)(const Int* m, const Int* n, const double* A, const Int* ia, 
    const Int* ja, const Int* desca, double* B,
    const Int* ib, const Int* jb, const Int* descb,
    const Int* contxt);

void SCALAPACK(pzgemr2d)(const Int* m, const Int* n, const dcomplex* A, const Int* ia, 
    const Int* ja, const Int* desca, dcomplex* B,
    const Int* ib, const Int* jb, const Int* descb,
    const Int* contxt);

// Trace of a square matrix
double SCALAPACK(pdlatra)(const Int *n , const double *a , const Int *ia , const Int *ja , const Int *desca );

// Cholesky
void SCALAPACK(pdpotrf)( const char* uplo, const Int* n, 
    double* A, const Int* ia, const Int* ja, const Int* desca, 
    Int* info );

void SCALAPACK(pdpotri)( const char* uplo, const Int* n, 
    double* A, const Int* ia, const Int* ja, const Int* desca, 
    Int* info );

// Eigenvalue problems
void SCALAPACK(pdsygst)( const Int* ibtype, const char* uplo, 
    const Int* n, double* A, const Int* ia, const Int* ja, 
    const Int* desca, const double* b, const Int* ib, const Int* jb,
    const Int* descb, double* scale, Int* info );

void SCALAPACK(pdlacpy)(const char* uplo,
    const Int* m, const Int* n,
    const double* A, const Int* ia, const Int* ja, const Int* desca, 
    const double* B, const Int* ib, const Int* jb, const Int* descb );

void SCALAPACK(pdsyev)(const char *jobz, const char *uplo, const Int *n, double *a, 
    const Int *ia, const Int *ja, const Int *desca, double *w, 
    double *z, const Int *iz, const Int *jz, const Int *descz, 
    double *work, const Int *lwork, Int *info);

void SCALAPACK(pdsyevd)(const char *jobz, const char *uplo, const Int *n, double *a, 
    const Int *ia, const Int *ja, const Int *desca, double *w, 
    const double *z, const Int *iz, const Int *jz, const Int *descz, 
    double *work, const Int *lwork, Int* iwork, const Int* liwork, 
    Int *info);

void SCALAPACK(pzheevd)(const char *jobz, const char *uplo, const Int *n, dcomplex *a, 
    const Int *ia, const Int *ja, const Int *desca, double *w, 
    dcomplex *z, const Int *iz, const Int *jz, const Int *descz, 
    dcomplex *work, const Int *lwork, double * rwork, int * lrwork,
    Int* iwork, Int* liwork, Int *info);

void SCALAPACK(pdsyevr)(const char *jobz, const char *range, const char *uplo,
    const Int *n, double* a, const Int *ia, const Int *ja,
    const Int *desca, const double* vl, const double *vu,
    const Int *il, const Int* iu, Int *m, Int *nz, 
    double *w, double *z, const Int *iz, const Int *jz, 
    const Int *descz, double *work, const Int *lwork, 
    Int *iwork, const Int *liwork, Int *info);


// Factorization and triangular solve
void SCALAPACK(pzgetrf)( const Int* m, const Int* n, dcomplex* A,
    const Int* ia, const Int* ja, const Int* desca, Int* ipiv,
    Int* info );

void SCALAPACK(pzgetri)( const Int* n, dcomplex* A, const Int* ia,
    const Int* ja, const Int* desca, const Int* ipiv, 
    dcomplex *work, const Int* lwork, Int *iwork, const Int *liwork, 
    Int* info );

// QRCP 
void SCALAPACK(pdgeqpf)( Int* m, Int* n, double* A, Int* ia, Int* ja,
    Int* desca, Int* ipiv, double* itau, double* work, Int* lwork, 
    Int* info ); 

// RQRCP by Jianwei Xiao, Julien Langou and Ming Gu
Int SCALAPACK(rqrcp)( Int *m, Int *n, Int *k, double *A, Int *descA, 
    Int *m_B, Int *n_B, double *B, Int *descB, double *OMEGA, 
    Int *desc_OMEGA, Int *ipiv, double *tau, Int *nb,
    Int *ipiv_a, double *tau_b, double *work, Int *lwork );

// Auxiliary routine used by RQRCP
void SCALAPACK(partial_pdgeqrf) ( int *m, int *n, int *k, double *a, int
    *ia, int *ja, int *descA, double *tau, double *work, int *lwork, int
    *info);


} //extern "C"

// *********************************************************************
// Interface for Descriptor
// *********************************************************************
class Descriptor{
private:
  std::vector<Int> values_;
  Int nprow_;
  Int npcol_;
  Int myprow_;
  Int mypcol_;
public:
  //in total 9 elements in the descriptor
  //NOTE: C convention is followed here!
  enum{
    DTYPE = 0,
    CTXT  = 1,
    M     = 2,
    N     = 3,
    MB    = 4,
    NB    = 5,
    RSRC  = 6,
    CSRC  = 7,
    LLD   = 8,
    DLEN  = 9, 
  };

  Descriptor() {values_.resize(DLEN); values_[CTXT] = -1;}

  /// @brief Constructor.
  ///
  /// NOTE: The leading dimension is directly computed rather than
  /// taking as an input.
  Descriptor(Int m, Int n, Int mb,
      Int nb, Int irsrc, Int icsrc,
      Int contxt) 
  {Init(m, n, mb, nb, irsrc, icsrc, contxt);}


  /// @brief Descriptor. Directly provide leading dimension
  Descriptor(Int m, Int n, Int mb,
      Int nb, Int irsrc, Int icsrc,
      Int contxt, Int lld) 
  {Init(m, n, mb, nb, irsrc, icsrc, contxt, lld);}


  ~Descriptor(){}

  /// @brief Initialize the descriptor. 
  /// NOTE: Currently ScaLAPACKMatrix requires the descriptor to be
  /// initialized only with this version.
  void Init(Int m, Int n, Int mb,
      Int nb, Int irsrc, Int icsrc,
      Int contxt);

  /// @brief Initialize the descriptor.  Directly provide lld
  void Init(Int m, Int n, Int mb,
      Int nb, Int irsrc, Int icsrc,
      Int contxt, Int lld);


  Int* Values() {return &values_[0];} 

  const Int* Values() const {return &values_[0];}

  Int NpRow() const {return nprow_;}

  Int NpCol() const {return npcol_;}

  Int MypRow() const {return myprow_;}

  Int MypCol() const {return mypcol_;}

  Int Get(Int i) const
  {
    if( i < 0 || i > DLEN ){
      std::ostringstream msg;
      msg << "Descriptor::Get takes value in [0,8]" << std::endl;
    }
    return values_[i];
  }


  Descriptor& operator=(const Descriptor& desc);
};


/*********************************************************************************
 * Interface for a ScaLAPACK Matrix
 *********************************************************************************/

template<typename F>
class ScaLAPACKMatrix{
private:
  Descriptor       desc_;
  std::vector<F>   localMatrix_;
public:

  // *********************************************************************
  // Lifecycle
  // *********************************************************************
  ScaLAPACKMatrix(){}

  ~ScaLAPACKMatrix(){}

  ScaLAPACKMatrix<F>& operator=(const ScaLAPACKMatrix<F>& A);


  /************************************************************
   * Basic information
   ************************************************************/

  Int Height() const {return desc_.Get(Descriptor::M);}

  Int Width() const {return desc_.Get(Descriptor::N);}

  Int MB()    const {return desc_.Get(Descriptor::MB);}

  Int NB()    const {return desc_.Get(Descriptor::NB);}

  Int Context() const {return desc_.Get(Descriptor::CTXT);}

  Int NumRowBlocks() const 
  {return (this->Height() + this->MB() - 1) / this->MB();}

  Int NumColBlocks() const
  {return (this->Width()  + this->NB() - 1) / this->NB();}

  // NOTE: LocalHeight is the same as LocalLDim here.
  Int LocalNumRowBlocks() const
  { return (this->NumRowBlocks() + this->desc_.NpRow() - 1 ) /
    this->desc_.NpRow(); }

  Int LocalNumColBlocks() const
  {return (this->NumColBlocks() + this->desc_.NpCol() - 1 ) /
    this->desc_.NpCol(); }

  Int LocalHeight() const 
  {return this->LocalNumRowBlocks() * this->MB(); }

  Int LocalWidth() const 
  {return this->LocalNumColBlocks() * this->NB(); }

  Int LocalLDim()  const 
  { 
    if( desc_.Get(Descriptor::LLD) != this->LocalHeight() )
    {
      std::ostringstream msg;
      msg 
        << "ScaLAPACK: the leading dimension does not match" << std::endl
        << "LLD from descriptor = " << desc_.Get(Descriptor::LLD) << std::endl
        << "LocalHeight         = " << this->LocalHeight() << std::endl;
    }
    return desc_.Get(Descriptor::LLD);
  }

  class Descriptor& Desc() {return desc_;}

  const class Descriptor&  Desc() const {return desc_;}

  F*  Data() {return &localMatrix_[0];}

  const F* Data() const {return &localMatrix_[0];}

  std::vector<F>&  LocalMatrix() {return localMatrix_;}

  /// @brief Change the descriptor of the matrix.
  ///
  /// NOTE: Changing the descriptor is the only way to resize the
  /// localMatrix.
  void SetDescriptor(const class Descriptor& desc)
  {
    desc_ = desc;
    localMatrix_.resize(this->LocalHeight() * this->LocalWidth());
  }

  /************************************************************
   * Entry manipulation
   ************************************************************/
  F  GetLocal( Int iLocal, Int jLocal ) const;

  void SetLocal( Int iLocal, Int jLocal, F val );
};


/*********************************************************************************
 * Methods
 *********************************************************************************/

/// @brief Redistribute a matrix A into a matrix B which shares the same
/// context as A. 
///
/// This routine performs p_gemr2d.

//void
//Gemm(char transA, char transB, const double alpha, 
//    const ScaLAPACKMatrix<double>& A, const ScaLAPACKMatrix<double>& B, 
//    const double beta, ScaLAPACKMatrix<double>& C );

void
Lacpy( char uplo,
    Int m, Int n,
    double* A, Int ia, Int ja, Int* desca,
    double* B, Int ib, Int jb, Int* descb);

void
Gemm( char transA, char transB,
    Int m, Int n, Int k,
    double alpha,
    double* A, Int ia, Int ja, Int* desca, 
    double* B, Int ib, Int jb, Int* descb,
    double beta,
    double* C, Int ic, Int jc, Int* descc,
    Int contxt);


void 
Syrk ( char uplo, char trans,
       Int n, int k,
       double alpha, 
       double *A, Int ia, Int ja, Int *desca,
       double beta, 
       double *C, Int ic, Int jc, Int *descc);

void
Syr2k (char uplo, char trans,
       Int n, int k,
       double alpha, 
       double *A, Int ia, Int ja, Int *desca,
       double *B, Int ib, Int jb, Int *descb,
       double beta,
       double *C, Int ic, Int jc, Int *descc);

void
Gemr2d(const ScaLAPACKMatrix<double>& A, ScaLAPACKMatrix<double>& B);


/// @brief Solve triangular matrix equation
///
/// @param[in]     A (local) Contains the local entries of the matrix A.
/// @param[in,out] B (local) Contains the local entries of the matrix B.
/// On exit, the local entries are overwritten by the solution.
void
Trsm( char side, char uplo, char trans, char diag, double alpha,
    const ScaLAPACKMatrix<double>& A, ScaLAPACKMatrix<double>& B );


/// @brief Compute the eigenvalues only for symmetric matrices. 
///
/// Performs p_syev.  
void
Syev(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs);

/// @brief Compute the eigenvalues and the eigenvectors for symmetric
/// matrices.  
///
/// Performs p_syev.  
/// NOTE: The eigenvector matrix Z is assumed to use the same
/// descriptor as A.
void
Syev(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z);

/// @brief Compute the eigenvalues and the eigenvectors for symmetric
/// matrices using the divide-and-conquer algorithm.  
///
/// Performs p_syevd.
/// NOTE: The eigenvector matrix Z is assumed to use the same
/// descriptor as A.
void
Syevd(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z);

void
Syevd(char uplo, ScaLAPACKMatrix<dcomplex>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<dcomplex>& Z);

/// @brief Compute the eigenvalues and the eigenvectors for symmetric
/// matrices using the MRRR algoritm for diagonalizing the tri-diagonal
/// problem.
///
/// Performs p_syevr.
/// NOTE: The eigenvector matrix Z is assumed to use the same
/// descriptor as A.
void 
Syevr(char uplo, ScaLAPACKMatrix<double>& A,
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z);

/// @brief Compute the selected range of eigenvalues and the
/// eigenvectors for symmetric matrices using the MRRR algoritm for
/// diagonalizing the tri-diagonal problem.
///
///
/// Performs p_syevr.
/// NOTE: The eigenvector matrix Z is assumed to use the same
/// descriptor as A.
///
///
/// @param[in] il The index (from smallest to largest) of the smallest
/// eigenvalue to be returned.  il >= 1 following the FORTRAN index
/// notation.
///
/// @param[in] iu The index (from smallest to largest) of the largest
/// eigenvalue to be returned.  iu >= 1 following the FORTRAN index
/// notation.
void 
Syevr(char uplo, ScaLAPACKMatrix<double>& A,
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z,
    Int il,
    Int iu);

/// @brief Compute the Cholesky factorization of an N-by-N 
/// real symmetric positive definite matrix.
///
/// @param[in,out] A (local) On entry, this array
/// contains the local pieces of the N-by-N symmetric distributed matrix
/// sub( A ) to be factored. On exit, if UPLO = 'U', the upper
/// triangular part of the distributed matrix contains the Cholesky
/// factor U, if UPLO = 'L', the lower triangular part of the
/// distributed matrix contains the Cholesky factor L. 
void 
Potrf( char uplo, ScaLAPACKMatrix<double>& A );


/// @brief Reduces a real symmetric-definite generalized eigenproblem
/// to standard form.
///
/// @param[in] ibtype (global) Matrix conversion type.  Currently always
/// set ibtype = 1, which overwrites A by inv(U^T)* A *inv(U) or
/// inv(L)*sub( A )*inv(L^T).
///
/// @param[in,out]  A (local) On entry, this array contains the local
/// pieces of the N-by-N symmetric distributed matrix A.  
/// On exit, the transformed matrix is stored in
/// the same format as A.
///
/// @param[in]      B (local) Contains the local pieces of the
/// triangular factor from the Cholesky factorization of B, as returned
/// by Potrf.
///
void 
Sygst( Int ibtype, char uplo, ScaLAPACKMatrix<double>& A,
    ScaLAPACKMatrix<double>& B );


/// @brief QRCP3 ScaLAPACK's verison of xGEQPF
void QRCPF( Int m, Int n, double* A, Int* desca, Int* piv, double* tau ); 


} // namespace scalapack
} // namespace dgdft

#endif // _SCALAPACK_HPP_

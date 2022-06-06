#include "scalapack.h"

namespace dgdft {
namespace scalapack {


// *********************************************************************
// Descriptor
// *********************************************************************

void
  Descriptor::Init(Int m, Int n, Int mb,
      Int nb, Int irsrc, Int icsrc,
      Int contxt )
  {
    values_.resize(DLEN);
    Cblacs_gridinfo(contxt, &nprow_, &npcol_, &myprow_, &mypcol_);

    // Compute the leading dimension.  Use the upper bound directly,
    // which costs a bit memory but saves the coding effort for book-keeping.
    Int lld = ( ( ( (m + mb - 1 ) / mb ) + nprow_ - 1 ) / nprow_ ) * mb;

    Int info;
    SCALAPACK(descinit)(&values_[0], &m, &n, &mb, &nb, &irsrc, &icsrc,
        &contxt, &lld, &info);
    if( info )
    {
      std::ostringstream msg;
      msg << "Descriptor:: descinit returned with info = " << info;
    }


    return ;
  }         // -----  end of method Descriptor::Init  ----- 


void
  Descriptor::Init(Int m, Int n, Int mb,
      Int nb, Int irsrc, Int icsrc,
      Int contxt, Int lld )
  {
    values_.resize(DLEN);
    Cblacs_gridinfo(contxt, &nprow_, &npcol_, &myprow_, &mypcol_);

    Int info;
    SCALAPACK(descinit)(&values_[0], &m, &n, &mb, &nb, &irsrc, &icsrc,
        &contxt, &lld, &info);
    if( info )
    {
      std::ostringstream msg;
      msg << "Descriptor:: descinit returned with info = " << info;
    }


    return ;
  }         // -----  end of method Descriptor::Init  ----- 


Descriptor& Descriptor::operator =    ( const Descriptor& desc  )
{
  if( this == &desc ) return *this;
  values_ = desc.values_;
  Cblacs_gridinfo(values_[CTXT], &nprow_, &npcol_, &myprow_, &mypcol_);
  if( nprow_ != desc.nprow_ ||
      npcol_ != desc.npcol_ ||
      myprow_ != desc.myprow_ ||
      mypcol_ != desc.mypcol_ ){
    std::ostringstream msg;
    msg << "Descriptor:: the context information does not match" << std::endl;
  }

  return *this;
}         // -----  end of method Descriptor::operator=  ----- 

// *********************************************************************
// ScaLAPACK routines
// *********************************************************************


template<class F>
inline F ScaLAPACKMatrix<F>::GetLocal    ( Int iLocal, Int jLocal ) const
{
  if( iLocal < 0 || iLocal > this->LocalHeight() ||
      jLocal < 0 || jLocal > this->LocalWidth() ){
    std::ostringstream msg;
    msg << "ScaLAPACK::GetLocal index is out of range" << std::endl;
  }

  return localMatrix_[iLocal + jLocal * (this->LocalHeight())];
}         // -----  end of method ScaLAPACKMatrix::GetLocal  ----- 


template<class F>
inline void ScaLAPACKMatrix<F>::SetLocal    ( Int iLocal, Int jLocal, F val )
{
  if( iLocal < 0 || iLocal > this->LocalHeight() ||
      jLocal < 0 || jLocal > this->LocalWidth() ){
    std::ostringstream msg;
    msg << "ScaLAPACK::SetLocal index is out of range" << std::endl;
  }
  localMatrix_[iLocal+jLocal*this->LocalHeight()] = val;

  return ;
}         // -----  end of method ScaLAPACKMatrix::SetLocal  ----- 


template<class F>
inline ScaLAPACKMatrix<F>& 
ScaLAPACKMatrix<F>::operator=    ( const ScaLAPACKMatrix<F>& A )
{
  if( this == &A ) return *this;
  desc_ = A.desc_;
  localMatrix_ = A.localMatrix_;
  return *this;
}         // -----  end of method ScaLAPACKMatrix::operator=  ----- 

// huwei
void
Lacpy( char uplo, 
    Int m, Int n, 
    double* A, Int ia, Int ja, Int* desca, 
    double* B, Int ib, Int jb, Int* descb){


  SCALAPACK(pdlacpy)( &uplo,
      &m, &n,
      A, &ia, &ja, desca, 
      B, &ib, &jb, descb );

  return;
}   // -----  end of function Lacpy  ----- 

// huwei
void
Gemm( char transA, char transB,
    Int m, Int n, Int k,
    double alpha,
    double* A, Int ia, Int ja, Int* desca, 
    double* B, Int ib, Int jb, Int* descb,
    double beta,
    double* C, Int ic, Int jc, Int* descc,
    Int contxt){


  SCALAPACK(pdgemm)( &transA, &transB,
      &m, &n, &k,
      &alpha,
      A, &ia, &ja, desca, 
      B, &ib, &jb, descb,
      &beta,
      C, &ic, &jc, descc);

  return;
}   // -----  end of function Gemm  ----- 

// Amartya Banerjee
void 
Syrk ( char uplo, char trans,
       Int n, int k,
       double alpha, 
       double *A, Int ia, Int ja, Int *desca,
       double beta, 
       double *C, Int ic, Int jc, Int *descc)
{
   SCALAPACK(pdsyrk)( &uplo, &trans, &n, &k,
		      &alpha, 
		      A, &ia, &ja, desca,
		      &beta, 
		      C, &ic, &jc, descc);
  
  return; 
  
}
       
void
Syr2k (char uplo, char trans,
       Int n, int k,
       double alpha, 
       double *A, Int ia, Int ja, Int *desca,
       double *B, Int ib, Int jb, Int *descb,
       double beta,
       double *C, Int ic, Int jc, Int *descc)
{
  SCALAPACK(pdsyr2k)(&uplo , &trans , 
		     &n , &k , 
		     &alpha , 
		     A , &ia , &ja , desca , 
		     B , &ib , &jb , descb , 
		     &beta , 
		     C , &ic , &jc , descc );

 
 return;    
}
       


void
Gemr2d(const ScaLAPACKMatrix<double>& A, ScaLAPACKMatrix<double>& B){
  if( A.Height() != B.Height() || A.Width() != B.Width() ){
    std::ostringstream msg;
    msg 
      << "Gemr2d:: Global matrix dimension does not match\n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl
      << "The dimension of B is " << B.Height() << " x " << B.Width() << std::endl;
  }

  if( A.Context() != B.Context() ){
    std::ostringstream msg;
    msg << "Gemr2d:: A and B are not sharing the same context." << std::endl; 
  }

  const Int M = A.Height();
  const Int N = A.Width();
  const Int contxt = A.Context();

  SCALAPACK(pdgemr2d)(&M, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), 
      B.Data(), &I_ONE, &I_ONE, 
      B.Desc().Values(), &contxt );    
  return;
}   // -----  end of function Gemr2d  ----- 



void
Trsm( char side, char uplo, char trans, char diag, double alpha,
    const ScaLAPACKMatrix<double>& A, ScaLAPACKMatrix<double>& B )
{
  const Int M = B.Height(); // const Int M = A.Height();
  const Int N = A.Width();

  SCALAPACK(pdtrsm)(&side, &uplo, &trans, &diag, &M, &N, &alpha,
      A.Data(), &I_ONE, &I_ONE, A.Desc().Values(), 
      B.Data(), &I_ONE, &I_ONE, B.Desc().Values());


  return ;
}        // -----  end of function Trsm  ----- 


void
Syev(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs){

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syev: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
  }

  char jobz = 'N';
  Int lwork = -1, info;
  std::vector<double> work(1);
  Int N = A.Height();

  eigs.resize(N);
  ScaLAPACKMatrix<double> dummyZ;

  SCALAPACK(pdsyev)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], dummyZ.Data(),
      &I_ONE, &I_ONE, dummyZ.Desc().Values(), &work[0],
      &lwork, &info);
  lwork = (Int)work[0];
  work.resize(lwork);

  SCALAPACK(pdsyev)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], dummyZ.Data(),
      &I_ONE, &I_ONE, dummyZ.Desc().Values(), &work[0],
      &lwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyev: logic error. Info = " << info << std::endl;
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyev: runtime error. Info = " << info << std::endl;
  }
  return;
}   // -----  end of function Syev ----- 



void
Syev(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z){
  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syev: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
  }

  char jobz = 'V';
  Int lwork = -1, info;
  std::vector<double> work(1);
  Int N = A.Height();

  eigs.resize(N);
  Z.SetDescriptor(A.Desc());

  SCALAPACK(pdsyev)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), &work[0],
      &lwork, &info);
  lwork = (Int)work[0];
  work.resize(lwork);

  SCALAPACK(pdsyev)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), &work[0],
      &lwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyev: logic error. Info = " << info << std::endl;
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyev: runtime error. Info = " << info << std::endl;
  }
  return;
}   // -----  end of function Syev ----- 

// FIXME here is memory issue in Syevd (lwork and liwork)
void
Syevd(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z){

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syevd: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
  }

  char jobz = 'V';
  Int  liwork = -1, lwork = -1, info;
  std::vector<double> work(1);
  std::vector<Int>    iwork(1);
  Int N = A.Height();

  eigs.resize(N);
  Z.SetDescriptor(A.Desc());

  SCALAPACK(pdsyevd)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&iwork[0], &liwork, &info);
  lwork = (Int)work[0];
  // NOTE: Buggy memory allocation in pdsyevd?
  lwork = lwork+2048;
  work.resize(lwork);
  liwork = iwork[0];
  // NOTE: Buggy memory allocation in pdsyevd?
  liwork = liwork+2048;
  iwork.resize(liwork);

  SCALAPACK(pdsyevd)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&iwork[0], &liwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevd: logic error. Info = " << info << std::endl;
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevd: runtime error. Info = " << info << std::endl;
  }
  return;
}   // -----  end of function Syevd ----- 

void
Syevd(char uplo, ScaLAPACKMatrix<dcomplex>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<dcomplex>& Z){

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syevd: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
  }

  char jobz = 'V';
  Int  liwork = -1, lwork = -1, lrwork = -1, info;
  std::vector<dcomplex> work(1);
  std::vector<Int>    iwork(1);
  std::vector<double> rwork(1);
  Int N = A.Height();

  eigs.resize(N);
  Z.SetDescriptor(A.Desc());

  SCALAPACK(pzheevd)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&rwork[0], &lrwork, &iwork[0], &liwork, &info);
  lwork = (Int)work[0].real();
  // NOTE: Buggy memory allocation in pdsyevd?
  lwork = lwork+2048;
  work.resize(lwork);
  liwork = iwork[0];
  // NOTE: Buggy memory allocation in pdsyevd?
  liwork = liwork+2048;
  iwork.resize(liwork);
  lrwork = (Int)rwork[0];
  rwork.resize(lrwork);

  SCALAPACK(pzheevd)(&jobz, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &eigs[0], Z.Data(),
      &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pzheevd: logic error. Info = " << info << std::endl;
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pzheevd: runtime error. Info = " << info << std::endl;
  }
  return;
}   // -----  end of function Syevd ----- 


void
Syevr(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z){

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syevr: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
  }

  char jobz = 'V';
  char range = 'A'; // Compute all eigenvalues

  Int  liwork = -1, lwork = -1, info;
  std::vector<double> work(1);
  std::vector<Int>    iwork(1);
  Int N = A.Height();

  eigs.resize(N);
  Z.SetDescriptor(A.Desc());
  double dummyV = 0.0;
  Int dummyI = 0;
  Int numEigValueFound, numEigVectorFound;


  SCALAPACK(pdsyevr)(&jobz, &range, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &dummyV, &dummyV, 
      &dummyI, &dummyI, &numEigValueFound, &numEigVectorFound,
      &eigs[0], Z.Data(), &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork, &iwork[0], &liwork, &info);
  lwork = (Int)work[0];
  work.resize(lwork);
  liwork = iwork[0];
  iwork.resize(liwork);

  SCALAPACK(pdsyevr)(&jobz, &range, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &dummyV, &dummyV, 
      &dummyI, &dummyI, &numEigValueFound, &numEigVectorFound,
      &eigs[0], Z.Data(), &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork, &iwork[0], &liwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevr: logic error. Info = " << info << std::endl;
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevr: runtime error. Info = " << info << std::endl;
  }

  if( numEigValueFound != N ){
    std::ostringstream msg;
    msg 
      << "pdsyevr: Not all eigenvalues are found.\n " 
      << "Found " << numEigValueFound << " eigenvalues, " << 
      N << " eigenvalues in total." << std::endl;
  }
  if( numEigVectorFound != N ){
    std::ostringstream msg;
    msg 
      << "pdsyevr: Not all eigenvectors are found.\n " 
      << "Found " << numEigVectorFound << " eigenvectors, " << 
      N << " eigenvectors in total." << std::endl;
  }
  return;
}   // -----  end of function Syevr ----- 

void
Syevr(char uplo, ScaLAPACKMatrix<double>& A, 
    std::vector<double>& eigs,
    ScaLAPACKMatrix<double>& Z,
    Int il,
    Int iu){

  if( A.Height() != A.Width() ){
    std::ostringstream msg;
    msg 
      << "Syevr: A must be a square matrix. \n" 
      << "The dimension of A is " << A.Height() << " x " << A.Width() << std::endl;
  }

  char jobz = 'V';
  char range = 'I'; // Compute selected range of eigenvalues

  Int  liwork = -1, lwork = -1, info;
  std::vector<double> work(1);
  std::vector<Int>    iwork(1);
  Int N = A.Height();
  Int numEigValue = std::min( N, iu - il + 1 );

  eigs.resize( N );
  Z.SetDescriptor(A.Desc());
  double dummyV = 0.0;
  Int dummyI = 0;
  Int numEigValueFound, numEigVectorFound;


  SCALAPACK(pdsyevr)(&jobz, &range, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &dummyV, &dummyV, 
      &il, &iu, &numEigValueFound, &numEigVectorFound,
      &eigs[0], Z.Data(), &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&iwork[0], &liwork, &info);
  lwork = (Int)work[0];
  work.resize(lwork);
  liwork = iwork[0];
  iwork.resize(liwork);

  SCALAPACK(pdsyevr)(&jobz, &range, &uplo, &N, A.Data(), &I_ONE, &I_ONE,
      A.Desc().Values(), &dummyV, &dummyV, 
      &il, &iu, &numEigValueFound, &numEigVectorFound,
      &eigs[0], Z.Data(), &I_ONE, &I_ONE, Z.Desc().Values(), 
      &work[0], &lwork,&iwork[0], &liwork, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevr: logic error. Info = " << info << std::endl;
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdsyevr: runtime error. Info = " << info << std::endl;
  }

  if( numEigValueFound != numEigValue ){
    std::ostringstream msg;
    msg 
      << "pdsyevr: Not all eigenvalues are found.\n " 
      << "Found " << numEigValueFound << " eigenvalues, " << 
      N << " eigenvalues in total." << std::endl;
  }
  if( numEigVectorFound != numEigValue ){
    std::ostringstream msg;
    msg 
      << "pdsyevr: Not all eigenvectors are found.\n " 
      << "Found " << numEigVectorFound << " eigenvectors, " << 
      N << " eigenvectors in total." << std::endl;
  }

  // Post processing of eigs by resize (not destroying the computed
  // eigenvalues) 
  eigs.resize( numEigValue );


  return;
}   // -----  end of function Syevr ----- 


void
Potrf( char uplo, ScaLAPACKMatrix<double>& A )
{
  Int info;

  Int N = A.Height();

  SCALAPACK(pdpotrf)(&uplo, &N, A.Data(), &I_ONE,
      &I_ONE, A.Desc().Values(), &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdpotrf: logic error. Info = " << info << std::endl;
  }
  else if( info > 0 )
  {
    std::ostringstream msg;
    msg << "pdpotrf: runtime error. Info = " << info << std::endl;
  }


  return ;
}     // -----  end of function Potrf  ----- 


void 
Sygst( Int ibtype, char uplo, ScaLAPACKMatrix<double>& A,
    ScaLAPACKMatrix<double>& B )
{

  Int info;
  double scale;
  Int N = A.Height();

  if( A.Height() != B.Height() ){
  }

  SCALAPACK(pdsygst)(&ibtype, &uplo, &N, A.Data(), 
      &I_ONE, &I_ONE, A.Desc().Values(),
      B.Data(), &I_ONE, &I_ONE, B.Desc().Values(), &scale, &info);

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "pdsygst: logic error. Info = " << info << std::endl;
  }


  return ;
}        // -----  end of function Sygst  ----- 


void QRCPF( Int m, Int n, double* A, Int* desca, Int* piv, double* tau) 
{
  if( m==0 || n==0 )
  {
    return;
  }

  Int lwork=-1, info;
  double dummyWork;
  int I_ONE = 1;

  SCALAPACK(pdgeqpf)(&m, &n, A, &I_ONE, &I_ONE, &desca[0],
      piv, tau, &dummyWork, &lwork, &info);

  lwork = dummyWork;
  std::vector<double> work(lwork);
  SCALAPACK(pdgeqpf)(&m, &n, A, &I_ONE, &I_ONE, &desca[0],
      piv, tau, &work[0], &lwork, &info);

  // Important: fortran index is 1-based. Change to 0-based
  for( Int i = 0; i < n; i++ ){
    piv[i]--;
  }

  if( info < 0 )
  {
    std::ostringstream msg;
    msg << "Argument " << -info << " had illegal value";
  }

  return;
}


} // namespace scalapack
} // namespace dgdft

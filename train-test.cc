#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include<sys/time.h>
#include<stdlib.h>
#include<unistd.h>

#include"scalapack.h"
#include"environment.h"

#include<mpi.h>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace dgdft::scalapack;

inline int IRound(double a){ 
  int b = 0;
  if(a>0) b = (a-int(a)<0.5)?int(a):(int(a)+1);
  else b = (int(a)-a<0.5)?int(a):(int(a)-1);
  return b; 
}

const int I_ZERO = 0;
const int I_ONE = 1;

extern "C" {

    void dpotrf_(char* UPLO, int* N, double* A, int *lda, int* INFO);

    void dtrsv_(char* UPLO, char* TRANSA, char* DIAG, int* N, double* A, int* lda, double* B, int* incb);

    void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* lda,
                    double* B, int* ldb, double* BETA, double* C, int* ldc);

    void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, double* A, int* lda, double* X, int* incx,
                    double* BETA, double* Y, int* incy);

    void daxpy_(int* N, double* ALPHA, double* X, int* incx, double* Y, int* incy);
    
}

int L = 6;
int pad = 2;
int pad_len = 2 * pad;
int channel = 1;
double J2 = 0.50;

struct proposeEntry{
    int source;
    int target;
};

struct proposePrime{
    int* spin_lattice;
    double energy;
    double ws;
    int index;
    int fly;
    std::vector<proposeEntry> propose_J1;
    std::vector<proposeEntry> propose_J2;
};

void batch_sprime(int* input_ptr, int batch_size, std::vector<proposePrime>& batchPrime) {
        
    batchPrime.clear();
    int* single_ptr = input_ptr;

    for(int i=0; i<batch_size; i++) {
        proposePrime singlePrime;
        singlePrime.spin_lattice = single_ptr;

        double energy = 0;
        int i_target, j_target;

        for(int i=0; i<L; i++){
            for(int j=0; j<L; j++){
                i_target = (i+1)%L;
                j_target = j;
                if (single_ptr[i*L+j]*single_ptr[i_target*L+j_target] == -1) {
                    proposeEntry pe;
                    pe.source = i*L+j;
                    pe.target = i_target*L+j_target;
                    singlePrime.propose_J1.push_back(pe);
                    energy -= 1;
                } else {
                    energy += 1;
                }

                i_target = i;
                j_target = (j+1)%L;
                if (single_ptr[i*L+j]*single_ptr[i_target*L+j_target] == -1) {
                    proposeEntry pe;
                    pe.source = i*L+j;
                    pe.target = i_target*L+j_target;
                    singlePrime.propose_J1.push_back(pe);
                    energy -= 1;
                } else {
                    energy += 1;
                }

                i_target = (i+1)%L;
                j_target = (j+1)%L;
                if (single_ptr[i*L+j]*single_ptr[i_target*L+j_target] == -1) {
                    proposeEntry pe;
                    pe.source = i*L+j;
                    pe.target = i_target*L+j_target;
                    singlePrime.propose_J2.push_back(pe);
                    energy -= J2;
                } else {
                    energy += J2;
                }

                i_target = (i+1)%L;
                j_target = (j-1+L)%L;
                if (single_ptr[i*L+j]*single_ptr[i_target*L+j_target] == -1) {
                    proposeEntry pe;
                    pe.source = i*L+j;
                    pe.target = i_target*L+j_target;
                    singlePrime.propose_J2.push_back(pe);
                    energy -= J2;
                } else {
                    energy += J2;
                }
            }
        }
        singlePrime.energy = energy;
        singlePrime.ws = 0;
        singlePrime.fly = 0;
        batchPrime.push_back(singlePrime);
        single_ptr += L*L;
    }
}

void PBC(int* input_ptr, int* output_ptr, int batch_size) {
    int* start_input = input_ptr;
    int* start_output = output_ptr;
    int j_target;
    int k_target;
    for(int i=0; i<batch_size; i++) {
        for(int j=0; j<L+2*pad; j++) {
            for(int k=0; k<L+2*pad;k++) {
                j_target = (j-pad+L)%L;
                k_target = (k-pad+L)%L;
                start_output[j*(L+2*pad)+k] = start_input[j_target*L+k_target];
            }
        }
        start_input += L*L;
        start_output += (L+2*pad)*(L+2*pad);
    }
}

void sign_rule(int* input_ptr, int batch_size, int* sign_result) {
    int* start = input_ptr;
    int count_positive;
    for(int i=0; i<batch_size; i++) {
        count_positive = 0;
        for(int j=0; j<L; j++)
            for(int k=0; k<L; k++)
                if((j%2==0 && k%2==0) || (j%2==1 && k%2==1)){
                    if (start[j*L+k]==1)
                        count_positive += 1;
            }
                
        if(count_positive%2 == 0) {
            sign_result[i] = 1;
        } else {
            sign_result[i] = -1;
        }
        start += L*L;
    }
}

int select_samples(double* Es_list, double* Os_list, int batch_size, int params_size, std::vector<proposePrime>& batchPrime, int* init_spin_lattice, int rank){
    int accept_samples_size = 0;
    int Es_size = 4*L*L;
    for(int i=0; i<batch_size; i++){
        if(Es_list[i]/Es_size<0 && Es_list[i]/Es_size>-10){
            Es_list[accept_samples_size] = Es_list[i];
            for(int j=0; j<params_size; j++)
                Os_list[accept_samples_size*params_size+j] = Os_list[i*params_size+j];
            accept_samples_size += 1;
            batchPrime[i].fly = 0;
        } else {
            batchPrime[i].fly += 1;

            if(batchPrime[i].fly >= 1){
                if (rank >= 0 && rank %16 == 0){
                    FILE *fp;
                    if((fp=fopen("ckpt/flyaway_record.txt", "a"))==NULL) {
                        printf("Cannot open file.\n");
                        exit(1);
                    }
                    fprintf(fp, "Rank: %d, chain_id: %d, ws: %.16e, Es: %.5f\n", rank, i, batchPrime[i].ws, Es_list[i]/Es_size);
                    fclose(fp);
                }

                for(int j=0; j<L; j++)
                    for(int k=0; k<L; k++)
                        batchPrime[i].spin_lattice[j*L+k] = init_spin_lattice[j*L+k];
                batchPrime[i].fly = 0;
            }
        }
    }
    return accept_samples_size;
}

void calculate_parameter(double* Es, double* Os, int batch_size, int params_size, double* Es_avg, double* Os_avg, double* OsEs_avg, int rank){
    
    double Es_sum = 0;
    for(int i=0; i<batch_size; i++)
        Es_sum += Es[i];
    *Es_avg = Es_sum/batch_size;

    for(int i=0; i<params_size; i++)
        Os_avg[i] = 0;
    for(int i=0; i<batch_size; i++)
        for(int j=0; j<params_size; j++)
            Os_avg[j] += Os[i*params_size+j];
    for(int i=0; i<params_size; i++)
        Os_avg[i] = Os_avg[i]/batch_size;

    double check_sum = 0;
    for(int i=0; i<params_size; i++)
        check_sum += Os_avg[i];

    //char TRANSA = 'N';
    //char TRANSB = 'T';
    double alpha = 1.0/batch_size;
    double beta = 0;
    int M = params_size;
    //int N = params_size;
    int K = batch_size;
    int lda = M;
    //int ldb = N;
    //int ldc = M;
    //double* OS_T = (double*)malloc(batch_size*params_size*sizeof(double));
    //for(int i=0; i<batch_size*params_size; i++)
    //    OS_T[i] = Os[i];
    //for(int i=0; i<params_size*params_size; i++)
    //    OO_avg[i] = 0;
    //dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, Os, &lda, OS_T, &ldb, &beta, OO_avg, &ldc);
    //free(OS_T);

    char TRANS = 'N';
    for(int i=0; i<params_size; i++)
        OsEs_avg[i] = 0;
    int incx = 1;
    int incy = 1;
    dgemv_(&TRANS, &M, &K, &alpha, Os, &lda, Es, &incx, &beta, OsEs_avg, &incy);

}

void compute_grad(double* Os_avg, double* Es_avg, double* OsEs_avg, double* dt, int params_size, double* grad){
    for(int i=0; i<params_size; i++)
        grad[i] = OsEs_avg[i];
    
    int N = params_size;
    double alpha = -1.0 * (*Es_avg);
    int incx = 1;
    int incy = 1;
    
    daxpy_(&N, &alpha, Os_avg, &incx, grad, &incy);
    for(int i=0; i<params_size; i++)
        grad[i] =  -1.0 * (*dt) * grad[i];
}

void covariance_matrix(double* OO_avg, double* Os_avg, double shift, int params_size){
    char TRANSA = 'N';
    char TRANSB = 'T';
    double alpha = -1.0;
    double beta = 1.0;
    int M = params_size;
    int N = params_size;
    int K = 1;
    int lda = M;
    int ldb = N;
    int ldc = M;
    double* Os_avg_T = (double*)malloc(params_size*sizeof(double));
    for(int i=0; i<params_size; i++)
        Os_avg_T[i] = Os_avg[i];
    dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, Os_avg, &lda, Os_avg_T, &ldb, &beta, OO_avg, &ldc);
    free(Os_avg_T);
    for(int i=0; i<params_size; i++)
        OO_avg[i*params_size+i] = OO_avg[i*params_size+i] + shift;
}

void compute_delta_scalapack(double* A, double* B, int N, double* C, int numProcess){

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numKeep = N;

    int contxt;
    int nprow, npcol, myrow, mycol, info;
    int numProcScaLAPACK = numProcess;

    for(int i = IRound(sqrt(double(numProcScaLAPACK))); i <= numProcScaLAPACK; i++) {
        nprow = i;
        npcol = int(numProcScaLAPACK / nprow);
        if (nprow * npcol == numProcScaLAPACK) break;
    }
    int scaBlockSize = int(numKeep/nprow);

    Cblacs_get(0, 0, &contxt);
    Cblacs_gridinit(&contxt, "C", nprow, npcol);
    Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);

    if(nprow>0 && npcol>0){

        int lda = numKeep;
        ScaLAPACKMatrix<double> square_mat_scala;
        Descriptor descReduceSeq, descReducePar;

        // Leading dimension provided
        descReduceSeq.Init(numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt, lda);

        // Automatically comptued Leading Dimension
        descReducePar.Init(numKeep, numKeep, scaBlockSize, scaBlockSize, I_ZERO, I_ZERO, contxt);

        square_mat_scala.SetDescriptor(descReducePar);

        //if (rank < 2)
        //    printf("nprow: %d, npcol: %d, start pdgemr2d\n", myrow, mycol);
        SCALAPACK(pdgemr2d)(&numKeep, &numKeep, A, &I_ONE, &I_ONE, descReduceSeq.Values(), 
                            &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt);
        //if (rank < 2)
        //    printf("nprow: %d, npcol: %d, finish pdgemr2d\n", myrow, mycol);

        SCALAPACK(pdpotrf)("L", &numKeep, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &info);
        //if (rank < 2)
        //    printf("pdpotrf OK\n");

        const int ht = square_mat_scala.Height();     
        // Set up v0
        dgdft::scalapack::Descriptor vec_desc;
        vec_desc.Init(ht, 1, scaBlockSize, scaBlockSize, 0, 0, contxt);   
        dgdft::scalapack::ScaLAPACKMatrix<double>  vec;
        vec.SetDescriptor(vec_desc);

        SCALAPACK(pdgemr2d)(&numKeep, &I_ONE, B, &I_ONE, &I_ONE, descReduceSeq.Values(), 
                            &vec.LocalMatrix()[0], &I_ONE, &I_ONE, vec.Desc().Values(), &contxt);
        //if (rank < 2)
        //    printf("pdgemr2d OK\n");

        SCALAPACK(pdtrsv)("L", "N", "N", &ht, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), 
                            vec.Data(), &I_ONE, &I_ONE, vec.Desc().Values(), &I_ONE);
        //if (rank < 2)
        //    printf("first pdtrsv OK\n");
        SCALAPACK(pdtrsv)("L", "T", "N", &ht, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), 
                            vec.Data(), &I_ONE, &I_ONE, vec.Desc().Values(), &I_ONE);
        //if (rank < 2)
        //    printf("second pdtrsv OK\n");

        for(int i=0; i<numKeep; i++)
            C[i] = 0;

        SCALAPACK(pdgemr2d)(&numKeep, &I_ONE, vec.Data(), &I_ONE, &I_ONE, vec.Desc().Values(),
                            C, &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt);
        //if (rank < 2)
        //    printf("pdgemr2d OK\n");

        Cblacs_gridexit(contxt);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void compute_delta(double* A, double* B, int N, double* C){
    int INFO;
    char UPLO = 'L';

    int num = N;

    for(int i=0; i<num; i++)
       C[i] = B[i];

    printf("start call dpotrf_\n");
    dpotrf_(&UPLO, &num, A, &num, &INFO);
    printf("finish call dpotrf_\n");

    char TRANSA = 'N';
    char DIAG = 'N';
    int lda = N;
    int incc = 1;
    printf("start call dtrsv_\n");
    dtrsv_(&UPLO, &TRANSA, &DIAG, &num, A, &lda, C, &incc);
    TRANSA = 'T';
    dtrsv_(&UPLO, &TRANSA, &DIAG, &num, A, &lda, C, &incc);
    printf("finish call dtrsv_\n");
}

void ReadTensorNames(const std::string filename, std::vector<std::string> &vars_list, std::vector<std::string> &grads_and_vars_list){

    std::ifstream ifile(filename);
    std::ostringstream buf;
    char ch;
    while(buf&&ifile.get(ch))
        buf.put(ch);
    std::string str = buf.str();

    int pos = 0;
    std::vector<std::string> grads, vars;
    int shift = 0;
    while(pos < str.length()){
        if(str[pos] == '\''){
            std::string temp;
            ++pos;
            while(pos < str.length() && str[pos] != '\''){
                temp += str[pos];
                ++pos;
            }
            if(shift == 0){
                grads.push_back(temp);
                shift = 1;
            } else {
                vars.push_back(temp);
                shift = 0;
            }
        }
        ++pos;
    }
    for(int i = 0; i < grads.size(); ++i)
        grads_and_vars_list.push_back(grads[i]);

    for(int i = 0; i < vars.size(); ++i){
        grads_and_vars_list.push_back(vars[i]);
        vars_list.push_back(vars[i]);
    }
}

int GetTotalLength(std::vector<tensorflow::Tensor>& out_tensors, int number){
    int total_length = 0;
    for(int i = 0; i < number; ++i)
        total_length += out_tensors[i].flat<double>().size();
    return total_length;
}

void AssignVars(std::vector<tensorflow::Tensor>& out_tensors, double* vars_buffer, int number){
    int idx_base = 0;
    for(int i = 0; i < number; ++i){
        auto vars_length = out_tensors[i].flat<double>().size();
        auto vars_ptr = out_tensors[i].flat<double>().data();
        for(int ii = 0; ii < vars_length; ++ii)
            vars_ptr[ii] = vars_buffer[idx_base + ii];
        idx_base += vars_length;
    }
}

void ReadVars(std::vector<tensorflow::Tensor>& out_tensors, double* vars_buffer, int number){
    int idx_base = 0;
    for(int i = 0; i < number; ++i){
        auto vars_length = out_tensors[i].flat<double>().size();
        auto vars_ptr = out_tensors[i].flat<double>().data();
        for (int ii = 0; ii < vars_length; ii++)
            vars_buffer[idx_base + ii] = vars_ptr[ii];
        idx_base += vars_length;
    }
}

void UpdateVars(std::vector<tensorflow::Tensor>& out_tensors, double* delta_buffer, int number){
    int idx_base = 0;
    for(int i = 0; i < number; ++i){
        auto vars_length = out_tensors[i].flat<double>().size();
        auto vars_ptr = out_tensors[i].flat<double>().data();
        for(int ii = 0; ii < vars_length; ++ii)
            vars_ptr[ii] = vars_ptr[ii] + delta_buffer[idx_base + ii];
        idx_base += vars_length;
    }
}

void ReadLogits(std::vector<tensorflow::Tensor>& out_tensors, double* logits){
    auto logits_ptr = out_tensors[0].flat<double>().data();
    auto logits_length = out_tensors[0].flat<double>().size();
    for(int i=0; i<logits_length; i++)
        logits[i] = logits_ptr[i];
}

void ReadGradsLogits(std::vector<tensorflow::Tensor>& out_tensors, double* grads_buffer, double* logits, int number){
    int idx_grad_base = 0;
    for(int i = 0; i < number; ++i){
        auto grads_length = out_tensors[i].flat<double>().size();
        auto grads_ptr = out_tensors[i].flat<double>().data();
        for (int ii = 0; ii < grads_length; ii++)
            grads_buffer[idx_grad_base + ii] = grads_ptr[ii];
        idx_grad_base += grads_length;
    }
    auto logits_ptr = out_tensors[number].flat<double>().data();
    auto logits_length = out_tensors[number].flat<double>().size();
    for(int i=0; i<logits_length; i++)
        logits[i] = logits_ptr[i];
}


void get_forward_batch(tensorflow::Tensor& forward_tensor, int batch_size, std::vector<proposePrime>& batchPrime){
    int* forward_spin_lattice = (int*)malloc(batch_size*L*L*sizeof(int));
    for (int i=0; i<batch_size; i++)
        for (int j=0; j<L; j++)
            for (int k=0; k<L; k++)
                forward_spin_lattice[i*L*L+j*L+k] = batchPrime[i].spin_lattice[j*L+k];    

    for (int i=0; i<batch_size; i++){
        int forward_index = rand() % batchPrime[i].propose_J1.size();
        //printf("i: %d, forward_index: %d\n", i, forward_index);
        batchPrime[i].index = forward_index;
        int index_source =  batchPrime[i].propose_J1[forward_index].source;
        int index_target = batchPrime[i].propose_J1[forward_index].target;
        forward_spin_lattice[i*L*L+index_source] = batchPrime[i].spin_lattice[index_target];
        forward_spin_lattice[i*L*L+index_target] = batchPrime[i].spin_lattice[index_source];
    }

    int* batch_PBC = (int*)malloc(batch_size*(L+2*pad)*(L+2*pad)*sizeof(int));
    PBC(forward_spin_lattice, batch_PBC, batch_size);
    free(forward_spin_lattice);

    auto forward_tensor_mapped = forward_tensor.tensor<double, 4>();
    for (int i=0; i<batch_size; i++)
        for (int j=0; j<(L+2*pad); j++)
            for (int k=0; k<(L+2*pad); k++)
                forward_tensor_mapped(i, 0, j, k) = double(batch_PBC[i*(L+2*pad)*(L+2*pad)+j*(L+2*pad)+k]);
    free(batch_PBC);          
}

void get_sprime_batch(tensorflow::Tensor& sprime_tensor, int batch_size, int total_sprime_size, std::vector<proposePrime>& batchPrime, int* sign_result){
    int* sprime_spin_lattice = (int*)malloc(total_sprime_size*L*L*sizeof(int));
    int* start_sprime = sprime_spin_lattice;
    for(int i=0; i<batch_size; i++){
        int J1_size = batchPrime[i].propose_J1.size();
        int J2_size = batchPrime[i].propose_J2.size();
        int sprime_size = J1_size+J2_size;
        for(int j=0; j<sprime_size; j++)
            for(int k=0; k<L; k++)
                for(int m=0; m<L; m++)
                    start_sprime[j*L*L+k*L+m] = batchPrime[i].spin_lattice[k*L+m];
        for(int j=0; j<J1_size; j++){
            int index_source =  batchPrime[i].propose_J1[j].source;
            int index_target = batchPrime[i].propose_J1[j].target;
            start_sprime[j*L*L+index_source] = batchPrime[i].spin_lattice[index_target];
            start_sprime[j*L*L+index_target] = batchPrime[i].spin_lattice[index_source];
        }
        for(int j=0; j<J2_size; j++){
            int index_source =  batchPrime[i].propose_J2[j].source;
            int index_target = batchPrime[i].propose_J2[j].target;
            start_sprime[(j+J1_size)*L*L+index_source] = batchPrime[i].spin_lattice[index_target];
            start_sprime[(j+J1_size)*L*L+index_target] = batchPrime[i].spin_lattice[index_source];
        }
        start_sprime += sprime_size*L*L;
    }

    sign_rule(sprime_spin_lattice, total_sprime_size, sign_result);

    int* batch_PBC = (int*)malloc(total_sprime_size*(L+2*pad)*(L+2*pad)*sizeof(int));
    PBC(sprime_spin_lattice, batch_PBC, total_sprime_size);
    free(sprime_spin_lattice);

    auto sprime_tensor_mapped = sprime_tensor.tensor<double, 4>();
    for (int j = 0; j < total_sprime_size; j++)
        for (int k = 0; k < (L+2*pad); k++)
            for (int m = 0; m < (L+2*pad); m++)
                sprime_tensor_mapped(j, 0, k, m) = double(batch_PBC[j*(L+2*pad)*(L+2*pad)+k*(L+2*pad)+m]);
    free(batch_PBC);
}

void get_sprime_batch_list(std::vector<tensorflow::Tensor> &split_sprime_tensor_list, int split_size, int batch_size, int total_sprime_size, std::vector<proposePrime>& batchPrime, int* sign_result){
    int* sprime_spin_lattice = (int*)malloc(total_sprime_size*L*L*sizeof(int));
    int* start_sprime = sprime_spin_lattice;
    for(int i=0; i<batch_size; i++){
        int J1_size = batchPrime[i].propose_J1.size();
        int J2_size = batchPrime[i].propose_J2.size();
        int sprime_size = J1_size+J2_size;
        for(int j=0; j<sprime_size; j++)
            for(int k=0; k<L; k++)
                for(int m=0; m<L; m++)
                    start_sprime[j*L*L+k*L+m] = batchPrime[i].spin_lattice[k*L+m];
        for(int j=0; j<J1_size; j++){
            int index_source =  batchPrime[i].propose_J1[j].source;
            int index_target = batchPrime[i].propose_J1[j].target;
            start_sprime[j*L*L+index_source] = batchPrime[i].spin_lattice[index_target];
            start_sprime[j*L*L+index_target] = batchPrime[i].spin_lattice[index_source];
        }
        for(int j=0; j<J2_size; j++){
            int index_source =  batchPrime[i].propose_J2[j].source;
            int index_target = batchPrime[i].propose_J2[j].target;
            start_sprime[(j+J1_size)*L*L+index_source] = batchPrime[i].spin_lattice[index_target];
            start_sprime[(j+J1_size)*L*L+index_target] = batchPrime[i].spin_lattice[index_source];
        }
        start_sprime += sprime_size*L*L;
    }

    sign_rule(sprime_spin_lattice, total_sprime_size, sign_result);

    int* batch_PBC = (int*)malloc(total_sprime_size*(L+2*pad)*(L+2*pad)*sizeof(int));
    PBC(sprime_spin_lattice, batch_PBC, total_sprime_size);
    free(sprime_spin_lattice);

    int repeat_times = total_sprime_size/split_size;
    int start_index = 0;
    for (int i=0; i<repeat_times; i++){
        tensorflow::Tensor split_spin_lattice_tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({split_size, channel, L+pad_len, L+pad_len}));
        auto split_spin_lattice_tensor_mapped = split_spin_lattice_tensor.tensor<double, 4>();
        for(int ii=0; ii<split_size; ii++)
            for (int j=0; j<(L+2*pad); j++)
                for (int k=0; k<(L+2*pad); k++)
                    split_spin_lattice_tensor_mapped(ii, 0, j, k) = double(batch_PBC[start_index+ii*(L+2*pad)*(L+2*pad)+j*(L+2*pad)+k]);
        split_sprime_tensor_list.push_back(split_spin_lattice_tensor);
        start_index += split_size*(L+2*pad)*(L+2*pad);
    }
    int tail_size = total_sprime_size%split_size;
    if(tail_size > 0){
        tensorflow::Tensor split_spin_lattice_tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({tail_size, channel, L+pad_len, L+pad_len}));
        auto split_spin_lattice_tensor_mapped = split_spin_lattice_tensor.tensor<double, 4>();
        for(int ii=0; ii<tail_size; ii++)
            for (int j=0; j<(L+2*pad); j++)
                for (int k=0; k<(L+2*pad); k++)
                    split_spin_lattice_tensor_mapped(ii, 0, j, k) = double(batch_PBC[start_index+ii*(L+2*pad)*(L+2*pad)+j*(L+2*pad)+k]);
        split_sprime_tensor_list.push_back(split_spin_lattice_tensor);
    }
    free(batch_PBC);
}

void get_spin_lattice_batch(tensorflow::Tensor& batch_spin_lattice_tensor, int batch_size, int* batch_spin_lattice){
    int* batch_PBC = (int*)malloc(batch_size*(L+2*pad)*(L+2*pad)*sizeof(int));
    PBC(batch_spin_lattice, batch_PBC, batch_size);


    auto batch_spin_lattice_tensor_mapped = batch_spin_lattice_tensor.tensor<double, 4>();
    for (int i=0; i<batch_size; i++)
        for (int j=0; j<(L+2*pad); j++)
            for (int k=0; k<(L+2*pad); k++)
                batch_spin_lattice_tensor_mapped(i, 0, j, k) = double(batch_PBC[i*(L+2*pad)*(L+2*pad)+j*(L+2*pad)+k]);
    free(batch_PBC);          
}

void get_backward_tensor_list(std::vector<tensorflow::Tensor> &split_spin_lattice_tensor_list, int batch_size, int* batch_spin_lattice, int* sign_result){
    sign_rule(batch_spin_lattice, batch_size, sign_result);
    split_spin_lattice_tensor_list.clear();
    int* batch_PBC = (int*)malloc(batch_size*(L+2*pad)*(L+2*pad)*sizeof(int));
    PBC(batch_spin_lattice, batch_PBC, batch_size);
    int split_size = 1;

    for (int i=0; i<batch_size; i++){
        tensorflow::Tensor split_spin_lattice_tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({split_size, channel, L+pad_len, L+pad_len}));
        auto split_spin_lattice_tensor_mapped = split_spin_lattice_tensor.tensor<double, 4>();
        for (int j=0; j<(L+2*pad); j++)
            for (int k=0; k<(L+2*pad); k++)
                split_spin_lattice_tensor_mapped(0, 0, j, k) = double(batch_PBC[i*(L+2*pad)*(L+2*pad)+j*(L+2*pad)+k]);
        split_spin_lattice_tensor_list.push_back(split_spin_lattice_tensor);
    }
    free(batch_PBC);

}

void update_batch_spin_lattice(std::vector<proposePrime>& batchPrime, int* batch_spin_lattice, int batch_size, double* ws_logits, int rank){
    double ws_new[batch_size];
    int fly_old[batch_size];
    
    for(int i=0; i<batch_size; i++){
        fly_old[i] = batchPrime[i].fly;
        //double r = 0;
        double r = (double(rand() % 1000))/1000;
        double P = ws_logits[i]/batchPrime[i].ws;
        int tmp;
        if(bool(P*P>r)){

            //if(i == 0){
            //    FILE *fp;
            //    if((fp=fopen("MC_record.txt", "a"))==NULL) {
            //        printf("Cannot open file.\n");
            //        exit(1);
            //    }

            //    fprintf(fp, "Rank: %d, chain_id: %d, P: %.16e, r: %.16e, ws_old: %.16e, ws_new: %.16e\n", rank, i, P*P, r, batchPrime[i].ws, ws_logits[i]);
            //    fclose(fp);
            //}

            int last_index = batchPrime[i].index;
            int index_source = batchPrime[i].propose_J1[last_index].source;
            int index_target = batchPrime[i].propose_J1[last_index].target;
            
            tmp = batchPrime[i].spin_lattice[index_source];
            batchPrime[i].spin_lattice[index_source] = batchPrime[i].spin_lattice[index_target];
            batchPrime[i].spin_lattice[index_target] = tmp;
            ws_new[i] = ws_logits[i];
        }else{
            ws_new[i] = batchPrime[i].ws;
        }
    }

    batch_sprime(batch_spin_lattice, batch_size, batchPrime);
    for(int i=0; i<batch_size; i++){
        batchPrime[i].ws = ws_new[i];
        batchPrime[i].fly = fly_old[i];
    }
}


void compute_OO_delta_scalapack(double* A, double* B, double* C, double shift, int batch_size, int total_length, int numProcess, double* delta){

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int contxt;
    int nprow, npcol, myrow, mycol, info;
    int numProcScaLAPACK = numProcess;

    for(int i = IRound(sqrt(double(numProcScaLAPACK))); i <= numProcScaLAPACK; i++) {
        nprow = i;
        npcol = int(numProcScaLAPACK / nprow);
        if (nprow * npcol == numProcScaLAPACK) break;
    }
    
    int numKeep_N = batch_size;
    int numKeep_M = total_length;
    //int scaBlockSize = int(numKeep/nprow);
    int scaBlockSize_M = int((numKeep_M+nprow-1)/nprow);
    int scaBlockSize_N = int((numKeep_N+npcol-1)/npcol);

    Cblacs_get(0, 0, &contxt);
    Cblacs_gridinit(&contxt, "C", nprow, npcol);
    Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);

    if(nprow>0 && npcol>0){

        struct timeval tv_start, tv_end;

        int lda = numKeep_M;
        ScaLAPACKMatrix<double> square_mat_scala_Os, square_mat_scala_OO;
        Descriptor descReduceSeq, descReducePar, descReduceSeq_OO, descReducePar_OO;

        descReduceSeq.Init(numKeep_M, numKeep_N, numKeep_M, numKeep_N, I_ZERO, I_ZERO, contxt, lda);
        descReducePar.Init(numKeep_M, numKeep_N, scaBlockSize_M, scaBlockSize_N, I_ZERO, I_ZERO, contxt);
        square_mat_scala_Os.SetDescriptor(descReducePar);

        descReduceSeq_OO.Init(numKeep_M, numKeep_M, numKeep_M, numKeep_M, I_ZERO, I_ZERO, contxt, lda);
        descReducePar_OO.Init(numKeep_M, numKeep_M, scaBlockSize_M, scaBlockSize_M, I_ZERO, I_ZERO, contxt);
        square_mat_scala_OO.SetDescriptor(descReducePar_OO);

        dgdft::scalapack::Descriptor vec_desc;
        vec_desc.Init(numKeep_M, 1, scaBlockSize_M, 1, I_ZERO, I_ZERO, contxt);   
        dgdft::scalapack::ScaLAPACKMatrix<double>  scala_grad, scala_delta;
        scala_grad.SetDescriptor(vec_desc);
        scala_delta.SetDescriptor(vec_desc);

        int single_batch_size = batch_size/size;
        int alltoall_block_length = int((total_length+npcol-1)/npcol);
        int alltoall_total_length = alltoall_block_length * npcol;

        double* alltoall_send_buf = (double*)malloc(single_batch_size*alltoall_total_length*sizeof(double));
        for(int i=0; i<single_batch_size; i++){
            for(int j=0; j<alltoall_total_length; j++){
                int target_index_i = j / alltoall_block_length;
                int target_index_j = (i*alltoall_block_length + j%alltoall_block_length) % (single_batch_size*alltoall_block_length);
                if(j<total_length)
                    alltoall_send_buf[target_index_i*(single_batch_size*alltoall_block_length)+target_index_j] = A[i*total_length+j];
                else
                    alltoall_send_buf[target_index_i*(single_batch_size*alltoall_block_length)+target_index_j] = 0;
            }
        }


        MPI_Comm col_comm;
        MPI_Comm_split(MPI_COMM_WORLD, rank / npcol, rank, &col_comm);
        double* alltoall_recv_buf = (double*)(&square_mat_scala_Os.LocalMatrix()[0]);
        MPI_Alltoall(alltoall_send_buf, single_batch_size*alltoall_block_length, MPI_DOUBLE, alltoall_recv_buf, single_batch_size*alltoall_block_length, MPI_DOUBLE, col_comm);
        free(alltoall_send_buf);

        //int start_index = scaBlockSize_M * rank;
        //for(int i=0; i<scaBlockSize_M; i++){
        //    if(start_index+i < total_length)
        //        scala_grad.LocalMatrix()[i] = B[start_index+i];
        //    else
        //        scala_grad.LocalMatrix()[i] = 0;
        //}
        //
        //for(int i=0; i<scaBlockSize_M; i++){
        //    if(start_index+i < total_length)
        //        scala_delta.LocalMatrix()[i] = C[start_index+i];
        //    else
        //        scala_delta.LocalMatrix()[i] = 0;
        //}

        //SCALAPACK(pdgemr2d)(&numKeep_M, &numKeep_N, A, &I_ONE, &I_ONE, descReduceSeq.Values(), 
        //                    &square_mat_scala_Os.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala_Os.Desc().Values(), &contxt);
        //
        SCALAPACK(pdgemr2d)(&numKeep_M, &I_ONE, B, &I_ONE, &I_ONE, descReduceSeq.Values(), 
                            &scala_grad.LocalMatrix()[0], &I_ONE, &I_ONE, scala_grad.Desc().Values(), &contxt);
        SCALAPACK(pdgemr2d)(&numKeep_M, &I_ONE, C, &I_ONE, &I_ONE, descReduceSeq.Values(), 
                            &scala_delta.LocalMatrix()[0], &I_ONE, &I_ONE, scala_delta.Desc().Values(), &contxt);

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&tv_start, NULL);

        double alpha = 1.0/numKeep_N;
        double beta = 0;
        SCALAPACK(pdgemm)("N", "T", &numKeep_M, &numKeep_M, &numKeep_N, 
            &alpha,
            square_mat_scala_Os.Data(), &I_ONE, &I_ONE, square_mat_scala_Os.Desc().Values(),
            square_mat_scala_Os.Data(), &I_ONE, &I_ONE, square_mat_scala_Os.Desc().Values(), 
            &beta,
            square_mat_scala_OO.Data(), &I_ONE, &I_ONE, square_mat_scala_OO.Desc().Values());
        
        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&tv_end, NULL);
        if(rank == 0)
            printf("pdgemm time: %f\n", double((tv_end.tv_sec-tv_start.tv_sec)+(tv_end.tv_usec-tv_start.tv_usec)/1000000.0));

        alpha = -1.0;
        beta = 1.0;
        int numCol = 1;
        SCALAPACK(pdgemm)("N", "T", &numKeep_M, &numKeep_M, &numCol, 
            &alpha,
            scala_grad.Data(), &I_ONE, &I_ONE, square_mat_scala_Os.Desc().Values(),
            scala_grad.Data(), &I_ONE, &I_ONE, square_mat_scala_Os.Desc().Values(), 
            &beta,
            square_mat_scala_OO.Data(), &I_ONE, &I_ONE, square_mat_scala_OO.Desc().Values());
        //printf("pdgemm OK\n");

        if(myrow==mycol){
            int local_height = square_mat_scala_OO.LocalHeight();
            int local_width = square_mat_scala_OO.LocalWidth();
            double* local_data = square_mat_scala_OO.Data();

            for(int i=0; i<local_height; i++)
                local_data[i*local_height+i] += shift;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&tv_start, NULL);

        SCALAPACK(pdpotrf)("L", &numKeep_M, square_mat_scala_OO.Data(), &I_ONE, &I_ONE, square_mat_scala_OO.Desc().Values(), &info);
        //printf("pdpotrf OK\n");
        
        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&tv_end, NULL);
        if(rank == 0){
            printf("pdpotrf time: %f, info: %d\n", double((tv_end.tv_sec-tv_start.tv_sec)+(tv_end.tv_usec-tv_start.tv_usec)/1000000.0), info);
        }
        if(info == 0){
            SCALAPACK(pdtrsv)("L", "N", "N", &numKeep_M, square_mat_scala_OO.Data(), &I_ONE, &I_ONE, square_mat_scala_OO.Desc().Values(), 
                                scala_delta.Data(), &I_ONE, &I_ONE, scala_delta.Desc().Values(), &I_ONE);
            SCALAPACK(pdtrsv)("L", "T", "N", &numKeep_M, square_mat_scala_OO.Data(), &I_ONE, &I_ONE, square_mat_scala_OO.Desc().Values(), 
                                scala_delta.Data(), &I_ONE, &I_ONE, scala_delta.Desc().Values(), &I_ONE);
            
            SCALAPACK(pdgemr2d)(&numKeep_M, &I_ONE, scala_delta.Data(), &I_ONE, &I_ONE, scala_delta.Desc().Values(),
                                delta, &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt);
            //printf("pdgemr2d OK\n");
            //for(int i=0; i<numKeep_M; i++)
            //    delta[i] = 0;
        } else {
            for(int i=0; i<numKeep_M; i++)
                delta[i] = 0;
        }

        Cblacs_gridexit(contxt);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    srand((unsigned)time(NULL)+rank);

    std::cout.precision(20);

    const string graph_def_filename = "graph_6x6_1w6_nosym.pb";
    const string init_model_prefix = "ckpt/model_restore";
    const string spin_lattice_prefix = "ckpt/spin_lattice_restore";

    std::unique_ptr<tensorflow::Session> session_;
    tensorflow::GraphDef graph_def;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_def_filename, &graph_def));
    session_.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_CHECK_OK(session_->Create(graph_def));

    TF_CHECK_OK(session_->Run({}, {}, {"init"}, nullptr));

    std::vector<std::string> grads_and_vars_list;
    std::vector<std::string> vars_list;
    std::string grads_and_vars_list_path = "grads_and_vars_1w6.txt";
    ReadTensorNames(grads_and_vars_list_path, vars_list, grads_and_vars_list);
    int number_vars = vars_list.size();
    std::vector<std::string> logits_list;
    logits_list.push_back("logits");

    std::vector<tensorflow::Tensor> vars_tensors;
    TF_CHECK_OK(session_->Run({}, {vars_list}, {}, &vars_tensors));

    int total_length = GetTotalLength(vars_tensors, number_vars);
    double *vars_buffer = (double*)malloc(total_length*sizeof(double));

    int* init_spin_lattice = (int*)malloc(L*L*sizeof(int));
    int batch_size = 512;
    int* batch_spin_lattice = (int*)malloc(batch_size*L*L*sizeof(int));

    int save_size = 5000;
    int restore_size = save_size;
    int* restore_batch_spin_lattice = (int*)malloc(restore_size*L*L*sizeof(int));
    int restore_step = 0;
    
    int* gather_buffer = (int*)malloc(size*L*L*sizeof(int));
    int* gather_Es_buffer = (int*)malloc(size*sizeof(double));
    FILE *fp;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank == 0){
        printf("model init OK\n");
        printf("Model total length: %d\n", total_length);
        printf("Matrix size: %ld\n", (long)total_length * total_length);
        printf("batch size: %d\n", batch_size);
    }

    if(rank == 0){
        if(argc > 1){
            if((fp=fopen("ckpt/good_init_9000", "r"))==NULL) {
                printf("Cannot open file.\n");
                exit(1);
            }
            fread(init_spin_lattice, sizeof(int), L*L, fp);
            fclose(fp);

            const string init_model_path = init_model_prefix + "_" + argv[1];
            if((fp=fopen(init_model_path.c_str(), "r"))==NULL) {
                printf("Cannot open file.\n");
                exit(1);
            }
            fread(vars_buffer, sizeof(double), total_length, fp);
            fclose(fp);

            const string init_spin_lattice_path = spin_lattice_prefix + "_" + argv[1];
            if((fp=fopen(init_spin_lattice_path.c_str(), "r"))==NULL) {
                printf("Cannot open file.\n");
                exit(1);
            }
            fread(restore_batch_spin_lattice, sizeof(int), restore_size*L*L, fp);
            fclose(fp);

            std::cout << "Restore model file: " << init_model_path << std::endl;
            std::cout << "Restore spin_lattice file: " << init_spin_lattice_path << std::endl;

        } else {
            TF_CHECK_OK(session_->Run({}, {vars_list}, {}, &vars_tensors));
            ReadVars(vars_tensors, vars_buffer, number_vars);
            //if((fp=fopen("ckpt/model_restore_0", "r"))==NULL) {
                //printf("Cannot open file.\n");
                //exit(1);
            //}
            //fread(vars_buffer, sizeof(double), total_length, fp);
            //fclose(fp);
        }
    }
    MPI_Bcast(vars_buffer, total_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    AssignVars(vars_tensors, vars_buffer, number_vars);

    if(argc > 1){
        restore_step = atoi(argv[1]);
        MPI_Bcast(init_spin_lattice, L*L, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(restore_batch_spin_lattice, restore_size*L*L, MPI_INT, 0, MPI_COMM_WORLD);
        if(restore_size <= batch_size){
            int start_batch_index = 0;
            for(int i=0; i<batch_size; i++)
                for(int j=0; j<L; j++)
                    for(int k=0; k<L; k++)
                        batch_spin_lattice[i*L*L+j*L+k] = restore_batch_spin_lattice[(i%restore_size)*L*L+j*L+k];
        } else {
            int start_batch_index = (rank * batch_size) % (restore_size - batch_size);
            for(int i=0; i<batch_size; i++)
                for(int j=0; j<L; j++)
                    for(int k=0; k<L; k++)
                        batch_spin_lattice[i*L*L+j*L+k] = restore_batch_spin_lattice[(start_batch_index+i)*L*L+j*L+k];
        }
    } else {
        for(int i=0; i<L; i++)
            for(int j=0; j<L; j++)
                if((i%2==0 && j%2==0) || (i%2==1 && j%2==1))
                    init_spin_lattice[i*L+j] = 1;
                else
                    init_spin_lattice[i*L+j] = -1;
        for(int i=0; i<batch_size; i++)
            for(int j=0; j<L; j++)
                for(int k=0; k<L; k++)
                    batch_spin_lattice[i*L*L+j*L+k] = init_spin_lattice[j*L+k];
    }

    double* logits_buffer = (double*)malloc(batch_size*sizeof(double));
    tensorflow::Tensor batch_spin_lattice_tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({batch_size, channel, L+pad_len, L+pad_len}));
    std::vector<tensorflow::Tensor> logits_tensors;
    get_spin_lattice_batch(batch_spin_lattice_tensor, batch_size, batch_spin_lattice); 
    TF_CHECK_OK(session_->Run({{"spin_lattice", batch_spin_lattice_tensor}}, {logits_list}, {}, &logits_tensors));
    ReadLogits(logits_tensors, logits_buffer);

    std::vector<proposePrime> target_batch_prime;
    batch_sprime(batch_spin_lattice, batch_size, target_batch_prime);
    for(int i=0; i<batch_size; i++)
        target_batch_prime[i].ws = logits_buffer[i];

    int step;
    int total_steps = 1000000;
    int warmup_step = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0 && argc == 1)
        printf("start warmup, warmup step: %d\n", warmup_step);

    tensorflow::Tensor forward_batch_tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({batch_size, channel, L+pad_len, L+pad_len}));
    if (argc == 1){
        if (rank == 0){
            /*
            if((fp=fopen("ckpt/good_init_spin_lattice", "r"))==NULL) {
                printf("Cannot open file.\n");
                exit(1);
            }
            fread(init_spin_lattice, sizeof(int), L*L, fp);
            fclose(fp);
            */
            
            for(step=0; step<warmup_step; step++){
                get_forward_batch(forward_batch_tensor, batch_size, target_batch_prime);
                TF_CHECK_OK(session_->Run({{"spin_lattice", forward_batch_tensor}}, {logits_list}, {}, &logits_tensors));
                ReadLogits(logits_tensors, logits_buffer);
                update_batch_spin_lattice(target_batch_prime, batch_spin_lattice, batch_size, logits_buffer, rank);
                if (step%100 == 0)
                    printf("step: %d, init_spin_lattice_ws: %.16e\n", step, target_batch_prime[0].ws);
                
                
                if (step%1000 == 500){
                    const string good_init_path = "ckpt/good_init_" + std::to_string(step);
                    for(int i=0; i<L; i++)
                        for(int j=0; j<L; j++)
                            init_spin_lattice[i*L+j] = target_batch_prime[0].spin_lattice[i*L+j];
                
                    if((fp=fopen(good_init_path.c_str(), "w"))==NULL) {
                        printf("Cannot open file.\n");
                        exit(1);
                    }
                    fwrite(init_spin_lattice, sizeof(int), L*L, fp);
                    fclose(fp);
                }
            }

            printf("init_spin_lattice_ws: %.16e\n", target_batch_prime[0].ws);
            for(int i=0; i<L; i++)
                for(int j=0; j<L; j++)
                    init_spin_lattice[i*L+j] = target_batch_prime[0].spin_lattice[i*L+j];
            
            if((fp=fopen("ckpt/good_init_spin_lattice", "w"))==NULL) {
                printf("Cannot open file.\n");
                exit(1);
            }
            fwrite(init_spin_lattice, sizeof(int), L*L, fp);
            fclose(fp);
            
            printf("finish warmup step: %d\n", step);
            //exit(0);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Bcast(init_spin_lattice, L*L, MPI_INT, 0, MPI_COMM_WORLD);
        for(int i=0; i<batch_size; i++)
            for(int j=0; j<L; j++)
                for(int k=0; k<L; k++)
                    batch_spin_lattice[i*L*L+j*L+k] = init_spin_lattice[j*L+k];

        get_spin_lattice_batch(batch_spin_lattice_tensor, batch_size, batch_spin_lattice); 
        TF_CHECK_OK(session_->Run({{"spin_lattice", batch_spin_lattice_tensor}}, {logits_list}, {}, &logits_tensors));
        ReadLogits(logits_tensors, logits_buffer);

        batch_sprime(batch_spin_lattice, batch_size, target_batch_prime);
        for(int i=0; i<batch_size; i++)
            target_batch_prime[i].ws = logits_buffer[i];
    }

    std::vector<std::string> grads_logits_list;
    for(int i=0; i<number_vars; i++)
        grads_logits_list.push_back(grads_and_vars_list[i]);
    grads_logits_list.push_back("logits");
    double dt = 0.01;
    double shift = 0.01;

    int* sign_batch_result = (int*)malloc(batch_size*sizeof(int));
    double* Os_list = (double*)malloc(batch_size*total_length*sizeof(double));
    double* Es_list = (double*)malloc(batch_size*sizeof(double));
    double* Os_list_com = (double*)malloc(batch_size*total_length*sizeof(double));
    double* Es_list_com = (double*)malloc(batch_size*sizeof(double));
    
    double* Os_avg = (double*)malloc(total_length*sizeof(double));
    double* Es_avg = (double*)malloc(sizeof(double));
    double* OsEs_avg = (double*)malloc(total_length*sizeof(double));

    double* first_order_grad_data = (double*)malloc(total_length*sizeof(double));
    double* delta = (double*)malloc(total_length*sizeof(double));

    struct timeval tv_start, tv_end;
    MPI_Barrier(MPI_COMM_WORLD);
    
    int start_train_step = 0;
    if(argc > 1)
        start_train_step = restore_step;
    else
        start_train_step = warmup_step;

    int com_size = 0;
    int target_com_size = 480;
    int internal_step = 5;
    int count_io = 1;
    int count_log = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
        printf("start_train_step: %d, target_com_size: %d, update_internal: %d, count_io: %d\n", start_train_step, target_com_size, internal_step, count_io);
    
    if(rank == 0){
        
        time_t rawtime;
        struct tm* timeinfo;
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        printf("Rank: %d, Size: %d, time: %s", rank, size, asctime(timeinfo));
    }

    for(step=start_train_step+1; step<total_steps; step++){
        get_forward_batch(forward_batch_tensor, batch_size, target_batch_prime);
        TF_CHECK_OK(session_->Run({{"spin_lattice", forward_batch_tensor}}, {logits_list}, {}, &logits_tensors));
        ReadLogits(logits_tensors, logits_buffer);
        update_batch_spin_lattice(target_batch_prime, batch_spin_lattice, batch_size, logits_buffer, rank);

        if(step%internal_step == 0){
            if(com_size < target_com_size){
                
                double* grads_ptr = Os_list;
                std::vector<tensorflow::Tensor> backward_tensor_list;
                get_backward_tensor_list(backward_tensor_list, batch_size, batch_spin_lattice, sign_batch_result);

                MPI_Barrier(MPI_COMM_WORLD);
                gettimeofday(&tv_start, NULL);
                
                for(int i=0; i<batch_size; i++){
                    std::vector<tensorflow::Tensor> grads_logits_tensors;
                    TF_CHECK_OK(session_->Run({{"spin_lattice", backward_tensor_list[i]}}, {grads_logits_list}, {}, &grads_logits_tensors));
                    ReadGradsLogits(grads_logits_tensors, grads_ptr, logits_buffer+i, number_vars);
                    double checksum;
                    for(int j=0; j<total_length; j++){
                        grads_ptr[j] = grads_ptr[j]/logits_buffer[i];
                    }
                    target_batch_prime[i].ws = logits_buffer[i];
                    checksum = 0;
                    for(int j=0; j<total_length; j++)
                        checksum += grads_ptr[j];
                    grads_ptr += total_length;
                    if(rank == 0 && i == 0){
                        printf("i: %d grads_checksum: %f\n", i, checksum);
                    }
                }
                
                MPI_Barrier(MPI_COMM_WORLD);
                gettimeofday(&tv_end, NULL);
                if(rank == 0){
                    printf("Os time: %f\n", double((tv_end.tv_sec-tv_start.tv_sec)+(tv_end.tv_usec-tv_start.tv_usec)/1000000.0));
                }

                gettimeofday(&tv_start, NULL);
                
                int total_sprime_size = 0;
                for(int i=0; i<batch_size; i++)
                    total_sprime_size = total_sprime_size + target_batch_prime[i].propose_J1.size() + target_batch_prime[i].propose_J2.size();

                int* sign_sprime_result = (int*)malloc(total_sprime_size*sizeof(int));
                double* logits_sprime_result = (double*)malloc(total_sprime_size*sizeof(double));

                std::vector<tensorflow::Tensor> sprime_batch_tensor_list;
                int split_size = 64;
                get_sprime_batch_list(sprime_batch_tensor_list, split_size, batch_size, total_sprime_size, target_batch_prime, sign_sprime_result);
                for(int i=0; i<sprime_batch_tensor_list.size(); i++){
                    TF_CHECK_OK(session_->Run({{"spin_lattice", sprime_batch_tensor_list[i]}}, {logits_list}, {}, &logits_tensors));
                    ReadLogits(logits_tensors, logits_sprime_result+split_size*i); 
                }
    
                int start_prime_index = 0;
                for(int i=0; i<batch_size; i++){
                    double J1_ws = 0;
                    double J2_ws = 0;
                    double sign_ws = target_batch_prime[i].ws * sign_batch_result[i];
                    int J1_size = target_batch_prime[i].propose_J1.size();
                    int J2_size = target_batch_prime[i].propose_J2.size();
                    for(int j=0; j<J1_size; j++)
                        J1_ws = J1_ws + logits_sprime_result[start_prime_index+j]*sign_sprime_result[start_prime_index+j]/sign_ws;
                    start_prime_index += J1_size;
                    for(int j=0; j<J2_size; j++)
                        J2_ws = J2_ws + logits_sprime_result[start_prime_index+j]*sign_sprime_result[start_prime_index+j]/sign_ws;
                    start_prime_index += J2_size;
                    Es_list[i] = target_batch_prime[i].energy + 2 * (J1_ws + J2*J2_ws);
                }
                if(rank == 0) {
                    FILE *fp;
                    if((fp=fopen("ckpt/Es_batch_size.txt", "a"))==NULL) {
                        printf("Cannot open file.\n");
                        exit(1);
                    }
                    for(int i=0; i<batch_size; i++){
                        fprintf(fp, "Rank: %d, Size: %d, chain_id: %d, ws: %.16e, Es: %.6f\n", rank, size, i, target_batch_prime[i].ws, Es_list[i]/(4*L*L));
                    }
                    fclose(fp);
                }
                
                free(sign_sprime_result);
                free(logits_sprime_result);

                MPI_Barrier(MPI_COMM_WORLD);
                gettimeofday(&tv_end, NULL);
                if(rank == 0){
                    printf("Es time: %f\n", double((tv_end.tv_sec-tv_start.tv_sec)+(tv_end.tv_usec-tv_start.tv_usec)/1000000.0));
                }
                
                gettimeofday(&tv_start, NULL);
                
                int accept_samples_size;
                accept_samples_size = select_samples(Es_list, Os_list, batch_size, total_length, target_batch_prime, init_spin_lattice, rank);

                MPI_Barrier(MPI_COMM_WORLD);
                gettimeofday(&tv_end, NULL);
                if(rank == 0){
                    printf("select_samples: %d, time: %f\n", accept_samples_size, double((tv_end.tv_sec-tv_start.tv_sec)+(tv_end.tv_usec-tv_start.tv_usec)/1000000.0));
                }
                int load_samples_size;
                if(accept_samples_size+com_size<target_com_size)
                    load_samples_size = accept_samples_size;
                else
                    load_samples_size = target_com_size - com_size;

                memcpy(Es_list_com+com_size, Es_list, load_samples_size*sizeof(double));
                memcpy(Os_list_com+com_size*total_length, Os_list, load_samples_size*total_length*sizeof(double));
                com_size += load_samples_size;
            }
            MPI_Barrier(MPI_COMM_WORLD);

            if(com_size == target_com_size){
                com_size = 0;

                gettimeofday(&tv_start, NULL);

                calculate_parameter(Es_list_com, Os_list_com, target_com_size, total_length, Es_avg, Os_avg, OsEs_avg, rank);

                MPI_Allreduce(MPI_IN_PLACE, Es_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, Os_avg, total_length, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, OsEs_avg, total_length, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                
                *Es_avg = *Es_avg/size;
                for(int i=0; i<total_length; i++)
                    Os_avg[i] = Os_avg[i]/size;
                for(int i=0; i<total_length; i++)
                    OsEs_avg[i] = OsEs_avg[i]/size;

                if(rank == 0){
                    
                    time_t rawtime;
                    struct tm* timeinfo;
                    time(&rawtime);
                    timeinfo = localtime(&rawtime);
                    FILE *fp;
                    if((fp=fopen("ckpt/Es_avg.txt", "a"))==NULL) {
                        printf("Cannot open file.\n");
                        exit(1);
                    }
                    fprintf(fp, "step: %d, Rank: %d, Size: %d, Es_avg: %f time: %s", step, rank, size, *Es_avg/(4*L*L), asctime(timeinfo));
                    fclose(fp);
                    
                    printf("step: %d, Rank: %d, Size: %d, Es_avg: %f, Os_avg: %f, time: %s", step, rank, size, *Es_avg/(4*L*L), Os_avg[0], asctime(timeinfo));
                }

                compute_grad(Os_avg, Es_avg, OsEs_avg, &dt, total_length, first_order_grad_data);
                MPI_Barrier(MPI_COMM_WORLD);
                gettimeofday(&tv_end, NULL);
                if(rank == 0){
                    printf("compute basic time: %f\n", double((tv_end.tv_sec-tv_start.tv_sec)+(tv_end.tv_usec-tv_start.tv_usec)/1000000.0));
                }

                gettimeofday(&tv_start, NULL);

                int numProc = size;
                int total_batch_size = target_com_size * size;
                compute_OO_delta_scalapack(Os_list_com, Os_avg, first_order_grad_data, shift, total_batch_size, total_length, numProc, delta);
                
                MPI_Barrier(MPI_COMM_WORLD);
                gettimeofday(&tv_end, NULL);
                if(rank == 0){
                    printf("compute scalapack time: %f\n", double((tv_end.tv_sec-tv_start.tv_sec)+(tv_end.tv_usec-tv_start.tv_usec)/1000000.0));
                }

                MPI_Bcast(delta, total_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                if(rank == 0){
                    double checksum = 0;
                    for(int i=0; i<10; i++)
                        std::cout << delta[i] << " ";
                    std::cout << std::endl;
                    for(int j=0; j<total_length; j++)
                        checksum += delta[j];
                    std::cout << "Rank: " << rank << " Size: " << size << " delta: " << checksum << std::endl;
                    std::cout << std::endl;
                }
                UpdateVars(vars_tensors, delta, number_vars);
                
                count_log++;
                if(count_log%count_io == 0){
                    count_log = 0;

                    double min = Es_list[0];
                    int min_id = 0;
                    for(int i=0; i<batch_size; i++){
                        if(Es_list[i] < min && Es_list[i]>*Es_avg * 1.0025){
                        //if((Es_list[i]/(4*L*L))<-0.494 && (Es_list[i]/(4*L*L))>-0.498){
                            min_id = i;
                            min = Es_list[i];
                        }
                    }

                    MPI_Gather(batch_spin_lattice+min_id*L*L, L*L, MPI_INT, gather_buffer, L*L, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Gather(&min, 1, MPI_DOUBLE, gather_Es_buffer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    if(size > save_size && rank == 0){
                        int select_count = 0;
                        double Es_max = *Es_avg * 0.9995;
                        double Es_min = *Es_avg * 1.0015;
                        for(int i=0; i<size; i++){
                            if(gather_Es_buffer[i] >= Es_min){
                                for(int j=0; j<L*L; j++)
                                    gather_buffer[select_count*L*L+j] = gather_buffer[i*L*L+j];
                                select_count++;
                            }
                        }
                        printf("IO spin_lattice select_count: %d, min: %f, max: %f\n", select_count, Es_min/(4*L*L), Es_max/(4*L*L));
                        if(select_count < save_size){
                            int repeat_times = save_size / select_count;
                            for(int j=1; j<repeat_times; j++){
                                for(int jj=0; jj<select_count; jj++)
                                    for(int jjj=0; jjj<L*L; jjj++)
                                        gather_buffer[(j*select_count+jj)*L*L+jjj] = gather_buffer[jj*L*L+jjj];
                            }
                            int tail_size = save_size - repeat_times*select_count;
                            int start_tail_index = save_size - tail_size;
                            for(int jj=0; jj<tail_size; jj++)
                                for(int jjj=0; jjj<L*L; jjj++)
                                    gather_buffer[(start_tail_index+jj)*L*L+jjj] = gather_buffer[jj*L*L+jjj];
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);                    
                    if(rank == 0){
                        TF_CHECK_OK(session_->Run({}, {vars_list}, {}, &vars_tensors));
                        ReadVars(vars_tensors, vars_buffer, number_vars);
                        const string init_model_path = init_model_prefix + "_" + std::to_string(step);
                        if((fp=fopen(init_model_path.c_str(), "w"))==NULL) {
                            printf("Cannot open file.\n");
                            exit(1);
                        }
                        fwrite(vars_buffer, sizeof(double), total_length, fp);
                        fclose(fp);

                        const string init_spin_lattice_path = spin_lattice_prefix + "_" + std::to_string(step);
                        if((fp=fopen(init_spin_lattice_path.c_str(), "w"))==NULL) {
                            printf("Cannot open file.\n");
                            exit(1);
                        }
                        //fwrite(gather_buffer, sizeof(int), size*L*L, fp);
                        fwrite(gather_buffer, sizeof(int), save_size*L*L, fp);
                        fclose(fp);
                    }
                }
            }
        }
    }
    free(restore_batch_spin_lattice);
    free(gather_buffer);
    free(vars_buffer);
    free(sign_batch_result);
    free(Es_list);
    free(Os_list);
    free(Es_list_com);
    free(Os_list_com);

    free(Es_avg);
    free(Os_avg);
    free(OsEs_avg);

    free(first_order_grad_data);
    free(delta);
    free(logits_buffer);
    free(init_spin_lattice);
    free(batch_spin_lattice);

    MPI_Finalize();

    return 0;
}

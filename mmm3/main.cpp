#include <iostream>
#include <unistd.h>
#include <omp.h>
#include <malloc.h>
#include <mkl.h> // used to test dgemm

using namespace std;

void test1() {
    cout << "Number of processors " 
         << sysconf(_SC_NPROCESSORS_ONLN) << '\n';
}

struct MMM_Info {   
    long A_dim_size[4];
    long B_dim_size[4];
    long C_dim_size[4];
    int  mul_dim1, mul_dim2;
};

/*
 *  Input: Matrices: A, B, C stored in dimentional order
 *         Matrices Info
 *  Ouput: Matrices C
 */
void d_4d_mm(double *A, double *B, double *C,
    long *A_dim_size, long *B_dim_size, long *C_dim_size, int mul_dim1, int mul_dim2)  {

    long i, j, k, l, m; // 1st, 2nd, ... dimentional size
    long A_index, B_index, C_index;    
    
    if (mul_dim1 == 1 && mul_dim2 == 2) { //
#pragma omp parallel for collapse(3) private(i, j, k, l, m, A_index, B_index, C_index)    
        for (i = 0; i < C_dim_size[0]; i++) {
            for (k = 0; k < C_dim_size[2]; k++) {
                for (l = 0; l < C_dim_size[3]; l++) {
                    for (m = 0; m < A_dim_size[mul_dim2-1]; m++) {
#pragma ivdep
#pragma omp simd
                        for (j = 0; j < C_dim_size[1]; j++) {
                            C_index = i*C_dim_size[1] + j
                                    + l*C_dim_size[0]*C_dim_size[1]
                                    + k*C_dim_size[0]*C_dim_size[1]*C_dim_size[3];
                            
                            A_index = i*A_dim_size[mul_dim2-1] + m 
                                    + l*A_dim_size[0]*A_dim_size[mul_dim2-1]
                                    + k*A_dim_size[0]*A_dim_size[mul_dim2-1]*A_dim_size[3]; 
                            
                            B_index = m*B_dim_size[1] + j 
                                    + l*A_dim_size[mul_dim2-1]*B_dim_size[1]
                                    + k*A_dim_size[mul_dim2-1]*B_dim_size[1]*B_dim_size[3];   
                            
                            C[C_index] += A[A_index] * B[B_index];
                       }
                    }
               }
            }
        }
    } else if (mul_dim1 == 1 && mul_dim2 == 3) { //
#pragma omp parallel for collapse(3) private(i, j, k, l, m, A_index, B_index, C_index)
        for (i = 0; i < C_dim_size[0]; i++) {
            for (k = 0; k < C_dim_size[2]; k++) {
                for (l = 0; l < C_dim_size[3]; l++) {
                    for (m = 0; m < A_dim_size[mul_dim2-1]; m++) {
#pragma ivdep
#pragma omp simd
                        for (j = 0; j < C_dim_size[1]; j++) {
                            C_index = i*C_dim_size[1] + j
                                    + l*C_dim_size[0]*C_dim_size[1]
                                    + k*C_dim_size[0]*C_dim_size[1]*C_dim_size[3];

                            A_index = i*A_dim_size[1] + j 
                                    + l*A_dim_size[0]*A_dim_size[1]
                                    + m*A_dim_size[0]*A_dim_size[1]*A_dim_size[3]; 

                            B_index = m*B_dim_size[1] + j 
                                    + l*A_dim_size[mul_dim2-1]*B_dim_size[1]
                                    + k*A_dim_size[mul_dim2-1]*B_dim_size[1]*B_dim_size[3];         
                                                        
                            C[C_index] += A[A_index] * B[B_index];
                       }
                    }
               }
            }
        }
    } else if (mul_dim1 == 1 && mul_dim2 == 4) { //
#pragma omp parallel for collapse(3) private(i, j, k, l, m, A_index, B_index, C_index)
        for (i = 0; i < C_dim_size[0]; i++) {
            for (k = 0; k < C_dim_size[2]; k++) {
                for (l = 0; l < C_dim_size[3]; l++) {
                    for (m = 0; m < A_dim_size[mul_dim2-1]; m++) {
#pragma ivdep
#pragma omp simd
                        for (j = 0; j < C_dim_size[1]; j++) {
                            C_index = i*C_dim_size[1] + j
                                    + l*C_dim_size[0]*C_dim_size[1]
                                    + k*C_dim_size[0]*C_dim_size[1]*C_dim_size[3];

                            A_index = i*A_dim_size[1] + j 
                                    + m*A_dim_size[0]*A_dim_size[1]
                                    + k*A_dim_size[0]*A_dim_size[1]*A_dim_size[mul_dim2-1]; 

                            B_index = m*B_dim_size[1] + j 
                                    + l*A_dim_size[mul_dim2-1]*B_dim_size[1]
                                    + k*A_dim_size[mul_dim2-1]*B_dim_size[1]*B_dim_size[3];        
                                                        
                            C[C_index] += A[A_index] * B[B_index];
                       }
                    }
               }
            }
        }
    } else if (mul_dim1 == 2 && mul_dim2 == 3) { //
#pragma omp parallel for collapse(3) private(i, j, k, l, m, A_index, B_index, C_index)
        for (i = 0; i < C_dim_size[0]; i++) {
            for (k = 0; k < C_dim_size[2]; k++) {
                for (l = 0; l < C_dim_size[3]; l++) {
                    for (m = 0; m < A_dim_size[mul_dim2-1]; m++) {
#pragma ivdep
#pragma omp simd
                        for (j = 0; j < C_dim_size[1]; j++) {
                            C_index = i*C_dim_size[1] + j
                                    + l*C_dim_size[0]*C_dim_size[1]
                                    + k*C_dim_size[0]*C_dim_size[1]*C_dim_size[3];
                                             
                            A_index = i*A_dim_size[1] + j 
                                    + l*A_dim_size[0]*A_dim_size[1]
                                    + m*A_dim_size[0]*A_dim_size[1]*A_dim_size[3]; 

                            B_index = i*A_dim_size[mul_dim2-1] + m 
                                    + l*B_dim_size[0]*A_dim_size[mul_dim2-1]
                                    + k*B_dim_size[0]*A_dim_size[mul_dim2-1]*B_dim_size[3];

                            C[C_index] += A[A_index] * B[B_index];
                       }
                    }
               }
            }
        }
    } else if (mul_dim1 == 2 && mul_dim2 == 4) { //
#pragma omp parallel for collapse(3) private(i, j, k, l, m, A_index, B_index, C_index)
        for (i = 0; i < C_dim_size[0]; i++) {
            for (k = 0; k < C_dim_size[2]; k++) {
                for (l = 0; l < C_dim_size[3]; l++) {
                    for (m = 0; m < A_dim_size[mul_dim2-1]; m++) {
#pragma ivdep
#pragma omp simd
                        for (j = 0; j < C_dim_size[1]; j++) {
                            C_index = i*C_dim_size[1] + j
                                    + l*C_dim_size[0]*C_dim_size[1]
                                    + k*C_dim_size[0]*C_dim_size[1]*C_dim_size[3];
                            
                            A_index = i*A_dim_size[1] + j 
                                    + m*A_dim_size[0]*A_dim_size[1]
                                    + k*A_dim_size[0]*A_dim_size[1]*A_dim_size[mul_dim2-1]; 

                            B_index = i*A_dim_size[mul_dim2-1] + m 
                                    + l*B_dim_size[0]*A_dim_size[mul_dim2-1]
                                    + k*B_dim_size[0]*A_dim_size[mul_dim2-1]*B_dim_size[3];

                            C[C_index] += A[A_index] * B[B_index];
                       }
                    }
               }
            }
        }
    } else { // mul_dim1 == 3 && mul_dim2 == 4
#pragma omp parallel for collapse(3) private(i, j, k, l, m, A_index, B_index, C_index)
        for (i = 0; i < C_dim_size[0]; i++) {
            for (k = 0; k < C_dim_size[2]; k++) {
                for (l = 0; l < C_dim_size[3]; l++) {
                    for (m = 0; m < A_dim_size[mul_dim2-1]; m++) {
#pragma ivdep
#pragma omp simd
                        for (j = 0; j < C_dim_size[1]; j++) {
                            C_index = i*C_dim_size[1] + j
                                    + l*C_dim_size[0]*C_dim_size[1]
                                    + k*C_dim_size[0]*C_dim_size[1]*C_dim_size[3];
                                                        
                            C[C_index] += A[A_index] * B[B_index];

                            A_index = i*A_dim_size[1] + j 
                                    + m*A_dim_size[0]*A_dim_size[1]
                                    + k*A_dim_size[0]*A_dim_size[1]*A_dim_size[mul_dim2-1]; 

                            B_index = i*B_dim_size[1] 
                                    + j + l*B_dim_size[0]*B_dim_size[1]
                                    + m*B_dim_size[0]*B_dim_size[1]*B_dim_size[3];
                       }
                    }
               }
            }
        }
    } 

}
void d4d_test1() {
    double A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // A = 1x2x2x3 
    double B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // B = 3x2x2x1
    double C[4];                                        // C = 1x2x2x1

    long A_dim_size[4];
    long B_dim_size[4];
    long C_dim_size[4];
    int  mul_dim1, mul_dim2;

    A_dim_size[0] = 1;  
    A_dim_size[1] = 2;  
    A_dim_size[2] = 2;  
    A_dim_size[3] = 3;  
    
    B_dim_size[0] = 3;  
    B_dim_size[1] = 2;  
    B_dim_size[2] = 2;  
    B_dim_size[3] = 1;  
    
    C_dim_size[0] = 1;      
    C_dim_size[1] = 2;      
    C_dim_size[2] = 2;      
    C_dim_size[3] = 1;
    
    mul_dim1 = 1;
    mul_dim2 = 4;      
    

    C[:] = 0;

    d_4d_mm(A, B, C, A_dim_size, B_dim_size, C_dim_size, mul_dim1, mul_dim2);

    for (int i = 0; i < 4; i++)
        cout << C[i] << ' ';
    cout << endl;
}

void d4d_test2() {
    double A[] = {1, 3, 5, 2, 4, 6, 13, 15, 17, 14, 16, 18, 
                    7, 9, 11, 8, 10, 12, 19, 21, 23, 20, 22, 24}; // A = 2x3x2x2
    double B[] = {24, 21, 23, 20, 22, 19, 12, 9, 11, 8, 10, 7,
                    18, 15, 17, 14, 16, 13, 6, 3, 5, 2, 4, 1}; // B = 3x2x2x2
    double C[16];                                        // C = 2x2x2x2

    long A_dim_size[4];
    long B_dim_size[4];
    long C_dim_size[4];
    int  mul_dim1, mul_dim2;

    A_dim_size[0] = 2;  
    A_dim_size[1] = 3;  
    A_dim_size[2] = 2;  
    A_dim_size[3] = 2;  
    
    B_dim_size[0] = 3;  
    B_dim_size[1] = 2;  
    B_dim_size[2] = 2;  
    B_dim_size[3] = 2;  
    
    C_dim_size[0] = 2;      
    C_dim_size[1] = 2;      
    C_dim_size[2] = 2;      
    C_dim_size[3] = 2;
    
    mul_dim1 = 1;
    mul_dim2 = 2;      
    

    C[:] = 0;

    d_4d_mm(A, B, C, A_dim_size, B_dim_size, C_dim_size, mul_dim1, mul_dim2);

    for (int i = 0; i < 16; i++)
        cout << C[i] << ' ';
    cout << endl;
}

void d3d_test1() {
    double A[] = {1, 2, 3, 4}; // A = 1x2x2x1
    double B[] = {1, 2, 3, 4}; // B = 2x1x2x1
    double C[2];                                        // C = 1x1x2x1

    long A_dim_size[4];
    long B_dim_size[4];
    long C_dim_size[4];
    int  mul_dim1, mul_dim2;

    A_dim_size[0] = 1;  
    A_dim_size[1] = 2;  
    A_dim_size[2] = 2;  
    A_dim_size[3] = 1;  
    
    B_dim_size[0] = 2;  
    B_dim_size[1] = 1;  
    B_dim_size[2] = 2;  
    B_dim_size[3] = 1;  
    
    C_dim_size[0] = 1;      
    C_dim_size[1] = 1;      
    C_dim_size[2] = 2;      
    C_dim_size[3] = 1;
    
    mul_dim1 = 1;
    mul_dim2 = 2;      
    

    C[:] = 0;

    d_4d_mm(A, B, C, A_dim_size, B_dim_size, C_dim_size, mul_dim1, mul_dim2);

    for (int i = 0; i < 2; i++)
        cout << C[i] << ' ';
    cout << endl;
}

void d4d_test3() {
    const long N = 100000000L;
#if 0
    double *A = new double[N]; // A = 100x100x100x100
    double *B = new double[N]; // B = 100x100x100x100
    double *C = new double[N]; // C = 100x100x100x100
#else
    double *A = (double*) _mm_malloc(sizeof(double)*N, 64); // A = 100x100x100x100
    double *B = (double*) _mm_malloc(sizeof(double)*N, 64); // B = 100x100x100x100
    double *C = (double*) _mm_malloc(sizeof(double)*N, 64); // C = 100x100x100x100
#endif 
    long A_dim_size[4];
    long B_dim_size[4];
    long C_dim_size[4];
    int  mul_dim1, mul_dim2;

    A_dim_size[0] = 100;  
    A_dim_size[1] = 100;  
    A_dim_size[2] = 100;  
    A_dim_size[3] = 100;  
    
    B_dim_size[0] = 100;  
    B_dim_size[1] = 100;  
    B_dim_size[2] = 100;  
    B_dim_size[3] = 100;  
    
    C_dim_size[0] = 100;      
    C_dim_size[1] = 100;      
    C_dim_size[2] = 100;      
    C_dim_size[3] = 100;
    
    mul_dim1 = 1;
    mul_dim2 = 2;      
    
    // Initialize
#pragma omp parallel for
    for (long i = 0; i < N; i++) {
        A[i] = i / 5.0;
        B[i] = 1; 
        C[i] = 0;
    }
    
    double elapsed_time, tmp = 0;
    for (int trial = 0; trial < 7; trial++) {
        elapsed_time = omp_get_wtime();
        d_4d_mm(A, B, C, A_dim_size, B_dim_size, C_dim_size, mul_dim1, mul_dim2);
        elapsed_time = omp_get_wtime() -  elapsed_time;
        if (trial > 2)  tmp += elapsed_time; 
        cout <<  "Time(s): " << elapsed_time << "\tGFLOPS: " 
             << (26.0 + 2*A_dim_size[mul_dim2-1])*N/1000000000.0/elapsed_time << endl;
    }
    cout << "#########\n";
    tmp /= 4.0;
    cout <<  "Average time(s): " << tmp  << "\tGFLOPS: "
         << (26.0 + 2*A_dim_size[mul_dim2-1])*N/1000000000.0/tmp << endl;

    /*
    for (int i = 0; i < 16; i++)
        cout << C[i] << ' ';
    cout << endl;
    */
#if 0
    delete []A;
    delete []B;
    delete []C;
#else
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
#endif
}

void dgemm_test() {
    const long N = 10000L;
    double *A = (double*) _mm_malloc(sizeof(double)*N*N, 64);
    double *B = (double*) _mm_malloc(sizeof(double)*N*N, 64);
    double *C = (double*) _mm_malloc(sizeof(double)*N*N, 64);

#pragma omp parallel for
    for (long i = 0; i < N*N; i++) {
        A[i] = i / 5.0;
        B[i] = 1;
        C[i] = 0;
    }

    double elapsed_time, tmp = 0;
    for (int trial = 0; trial < 7; trial++) {
        elapsed_time = omp_get_wtime();
        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        N, N, N, 1.0, A, N, B, N, 0.0, C, N);
        
        elapsed_time = omp_get_wtime() -  elapsed_time;
        if (trial > 2)  tmp += elapsed_time;
        cout <<  "Time(s): " << elapsed_time << endl;
    }


    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
}

int main() {
	#ifdef __MIC__ 
        cout << "MIC\n";
    #endif
    d4d_test1();
	//d4d_test2();
	//d3d_test1();
	dgemm_test();
    //d4d_test3();
    return 0;
}

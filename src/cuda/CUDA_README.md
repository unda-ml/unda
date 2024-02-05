## This is the folder where we will implement all of the CUDA functions for Matrix math

### Here's the general structure of how that will look in rust:

When we create a CUDA function in the .cu file, we can expose the endpoint via adding:

```c
extern "C" {
  void foo(){...}
}
```

## C Function Schema

All of the C functions we create should take in only pointers, we can create C structs that model Rust structs and send a Rust pointer to the C function and it will assume the shape of the C struct(in rust we have to convert all primitive types to the respective C variants). There should not be a return type rather a result pointer should always be made in Rust and passed through the C function, mutating it there. This will allow for the Rust code to interface with this resultant variable through dereferencing the pointer.

### Example:

## C:
```c
extern "C" {
    void cuda_matrix_mul(float *result, const CMatrix *matrix_a, const CMatrix *matrix_b){
        //Transfer to device mem :)
        float *result_dev, *matrix_a_dev, *matrix_b_dev;

        cudaMalloc((void**)&result_dev, matrix_a->rows * matrix_b->cols * sizeof(float));

        cudaMalloc((void**)&matrix_a_dev, matrix_a->len*sizeof(float));
        cudaMalloc((void**)&matrix_b_dev, matrix_b->len*sizeof(float));

        //Copy matrix A and B to A-device and B-device memory allocations
        cudaMemcpy(matrix_a_dev, matrix_a->data, matrix_a->len * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix_b_dev, matrix_b->data, matrix_b->len * sizeof(float), cudaMemcpyHostToDevice);

        //Split our grids and block sizes into blocks of 16x16
        dim3 blockSize(16,16);
        dim3 gridSize((matrix_b->cols + blockSize.x - 1) / blockSize.x, (matrix_a->rows + blockSize.y - 1) / blockSize.y);
        //Call cuda function!
        matrixMul<<<gridSize, blockSize>>>(result_dev, matrix_a_dev, matrix_b_dev, matrix_a->rows, matrix_a->cols, matrix_b->cols); 

        cudaError_t err = cudaMemcpy(result, result_dev, matrix_a->rows * matrix_b->cols * sizeof(float), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        
        cudaFree(result_dev);
        cudaFree(matrix_a_dev);
        cudaFree(matrix_b_dev);
    }
}
```

## Rust:
```rust
#[link(name = "matrix_math", kind="static")]
extern "C"{
    fn cuda_matrix_mul(result: *mut f32, mat_a: *const CMatrix, mat_b: *const CMatrix);
}
#[derive(Debug)]
#[repr(C)]
pub struct CMatrix{
    pub rows: c_int,
    pub col: c_int,
    pub data: * const c_float,
    pub len: c_int
}

impl CMatrix{
    pub fn new(rows: usize, cols: usize, data: &Vec<f32>) -> CMatrix{
        CMatrix { rows: rows as i32, col: cols as i32, data: data.as_ptr(), len: data.len() as i32 }
    }
}

impl ops::Mul<&Matrix> for &Matrix{
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix{
        if self.columns != other.rows{
            panic!("Matrix multiplication is in invalid format");
        }
        //println!("{:?}", self.par_multiply(other).data);
        let mut result: Vec<f32> = vec![0.0; (self.rows * other.columns) as usize];
        unsafe{
            cuda_matrix_mul(result.as_mut_ptr(), &self.to_cmatrix() as *const CMatrix, &other.to_cmatrix() as *const CMatrix);
            //println!("{:?}", result);
            return Matrix::from_flat(self.rows, other.columns, result);
        }
    }
}
```

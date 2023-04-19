use std::vec::Vec;
extern crate time;
use time::Instant;
#[macro_use]
extern crate arrayref;
extern crate rand;
extern crate rand_distr;
use rand::{distributions::Distribution, thread_rng};
use rand_distr::{Float, Normal};

extern crate rayon;
use rayon::prelude::*;
extern crate cblas;
extern crate intel_mkl_src;
use cblas::{ddot, Layout, Transpose, dgemv};

extern "C" {
    pub fn vdAdd(n: ::std::os::raw::c_int, a: *const f64, b: *const f64, r: *mut f64);
}

#[inline]
fn fill_vec_with_random_dist<'a, T: Float + Send + Sync, D: Distribution<T> + Sync>(
    vec: &mut Vec<T>,
    distribution: &'a D,
) {
    vec.par_iter_mut().for_each_init(
        || thread_rng(),
        |rng, x| {
            *x = distribution.sample(rng);
        },
    );
}

struct Timer {
    record_time: Instant,
}

impl Timer {
    fn tic(&mut self) {
        self.record_time = Instant::now();
    }

    fn toc(&mut self, info: &str) {
        println!(
            "{}{} milliseconds",
            info,
            (Instant::now() - self.record_time).whole_microseconds() as f64 / 1000f64
        );
    }
}

fn main() {
    // rayon::ThreadPoolBuilder::new().num_threads(8).build_global().unwrap();

    let mut timer = Timer {
        record_time: Instant::now(),
    };
    let normal = Normal::new(0f64, 3f64).unwrap();
    const N: usize = 210_000_000;
    println!("The size of vector is {:?}", N);
    let n: i32 = N as i32;

    let mut x: Vec<f64> = vec![0f64; N];
    timer.tic();
    fill_vec_with_random_dist(&mut x, &normal);
    timer.toc("The time of generating random normal number of x: ");

    let mut y: Vec<f64> = vec![0f64; N];
    timer.tic();
    fill_vec_with_random_dist(&mut y, &normal);
    timer.toc("The time of generating random normal number of y: ");
    println!("x: {:?}, {:?}", x[0], x[1000]);
    println!("y: {:?}, {:?}", y[0], y[1000]);

    // test on ddot
    timer.tic();
    let res = unsafe { ddot(n, &x, 1, &y, 1) };
    timer.toc("The time of Intel MKL cblas_ddot: ");
    println!("Result: {:?}", res);

    timer.tic();
    let res: f64 = x.par_iter().zip(y.par_iter()).map(|(a, b)| a * b).sum();
    timer.toc("The time of rayon ddot: ");
    println!("Result: {:?}", res);

    // test on vdAdd
    let xa = array_ref!(x, 0, N);
    let ya = array_ref!(y, 0, N);
    let mut r: Vec<f64> = vec![0f64; N];
    let ra = array_mut_ref!(r, 0, N);
    timer.tic();
    unsafe { vdAdd(n, xa.as_ptr(), ya.as_ptr(), ra.as_mut_ptr()) };
    timer.toc("The time of Intel MKL vdAdd: ");
    println!("Result: {:?}, {:?}, {:?}", r[0], r[1], r[2]);

    timer.tic();
    let r: Vec<f64> = x.par_iter().zip(y.par_iter()).map(|(a, b)| a + b).collect();
    timer.toc("The time of rayon vdAdd: ");
    println!("Result: {:?}, {:?}, {:?}", r[0], r[1], r[2]);

    // test on dgemv
    let (m, k) = (6000, 200000);
    let alpha: f64 = 2.0;
    let beta: f64 = 1.0;
    let mut a: Vec<f64> = vec![0f64; m*k];
    timer.tic();
    fill_vec_with_random_dist(&mut a, &normal);
    timer.toc(&format!("The time of generating random normal number of a ({0}x{1}): ", m, k));

    let mut b: Vec<f64> = vec![0f64; k];
    timer.tic();
    fill_vec_with_random_dist(&mut b, &normal);
    timer.toc(&format!("The time of generating random normal number of a ({0}x1): ", k));

    let mut c: Vec<f64> = vec![2f64; m];
    timer.tic();
    unsafe {
        dgemv(Layout::RowMajor, Transpose::None,
              m as i32, k as i32, alpha, &a, k as i32, &b, 1i32, beta, &mut c, 1i32);
    }
    timer.toc("The time of cblas_dgemv: ");
    println!("Result: {:?}, {:?}, {:?}", c[0], c[1], c[2]);

    let mut c: Vec<f64> = vec![2f64; m];
    timer.tic();
    a.par_chunks(k as usize)
     .zip(c.par_iter_mut())
     .for_each(|(chunk, r)| {
        *r = alpha * chunk.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>() + beta * *r;
     });
    timer.toc("The time of rayon: ");
    println!("Result: {:?}, {:?}, {:?}", c[0], c[1], c[2]);
}

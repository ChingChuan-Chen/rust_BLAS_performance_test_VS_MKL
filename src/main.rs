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
#[cfg(all(feature = "nightly", feature = "packed_simd"))]
extern crate packed_simd;
use packed_simd::f64x8;
extern crate cblas;
extern crate intel_mkl_src;
use cblas::ddot;

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
    let mut timer = Timer {
        record_time: Instant::now(),
    };
    let normal = Normal::new(0f64, 3f64).unwrap();
    const N: usize = 300_000_000;
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

    timer.tic();
    let res = unsafe { ddot(n, &x, 1, &y, 1) };
    timer.toc("The time of Intel MKL cblas_ddot: ");
    println!("Result: {:?}", res);

    timer.tic();
    let res: f64 = x.par_iter().zip(y.par_iter()).map(|(a, b)| a * b).sum();
    timer.toc("The time of rayon ddot: ");
    println!("Result: {:?}", res);

    timer.tic();
    let res: f64 = x
        .par_chunks(8)
        .map(f64x8::from_slice_unaligned)
        .zip(y.par_chunks(8).map(f64x8::from_slice_unaligned))
        .map(|(a, b)| (a * b).sum())
        .sum();
    timer.toc("The time of rayon + SIMD ddot: ");
    println!("Result: {:?}", res);

    let xa = array_ref!(x, 0, N);
    let ya = array_ref!(y, 0, N);
    let mut r: Vec<f64> = vec![0f64; N];
    let mut ra = array_mut_ref!(r, 0, N);
    timer.tic();
    unsafe { vdAdd(n, xa.as_ptr(), ya.as_ptr(), ra.as_mut_ptr()) };
    timer.toc("The time of Intel MKL vdAdd: ");
    println!("Result: {:?}, {:?}, {:?}", r[0], r[1], r[2]);

    timer.tic();
    let r: Vec<f64> = x.par_iter().zip(y.par_iter()).map(|(a, b)| a + b).collect();
    timer.toc("The time of rayon vdAdd: ");
    println!("Result: {:?}, {:?}, {:?}", r[0], r[1], r[2]);

    let mut r: Vec<f64> = vec![0f64; N];
    timer.tic();
    x.par_chunks(8)
        .map(f64x8::from_slice_unaligned)
        .zip(y.par_chunks(8).map(f64x8::from_slice_unaligned))
        .map(|(a, b)| a + b)
        .zip(r.par_chunks_mut(8))
        .for_each(|(v, slice)| {
            v.write_to_slice_unaligned(slice);
        });
    timer.toc("The time of rayon + SIMD vdAdd: ");
    println!("Result: {:?}, {:?}, {:?}", r[0], r[1], r[2]);
}

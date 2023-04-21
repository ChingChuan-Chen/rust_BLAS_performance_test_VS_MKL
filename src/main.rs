#![feature(array_chunks)]
#![feature(slice_as_chunks)]
#![feature(portable_simd)]
use std::simd::f64x4;
use std::simd::SimdFloat;
use std::time::Instant;
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
extern crate intel_mkl_sys;
extern crate num_cpus;
use cblas::{ddot, dgemv, Layout, Transpose};
use intel_mkl_sys::vdAdd;
extern crate packed_simd;

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
            (Instant::now() - self.record_time).as_micros() as f64 / 1000f64
        );
    }
}

fn dot_product_packed_simd(x: &[f64], y: &[f64]) -> f64 {
    let len = x.len();
    assert_eq!(len, y.len());

    let mut sum: packed_simd::Simd<[f64; 4]> = packed_simd::f64x4::splat(0.0);
    let chunk_size = packed_simd::f64x4::lanes();
    let num_chunks = len / chunk_size;

    for i in 0..num_chunks {
        let xi = packed_simd::f64x4::from_slice_unaligned(&x[i * chunk_size..]);
        let yi = packed_simd::f64x4::from_slice_unaligned(&y[i * chunk_size..]);
        sum += xi * yi;
    }

    let mut res = 0.0;
    for i in 0..chunk_size {
        res += sum.extract(i);
    }

    for i in num_chunks * chunk_size..len {
        res += x[i] * y[i];
    }

    res
}

fn dot_product_rayon_packed_simd(x: &[f64], y: &[f64]) -> f64 {
    let num_chunks: usize = rayon::current_num_threads();
    let chunk_size = x.len() / num_chunks;

    let dot_products: Vec<f64> = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            let start = i * chunk_size;
            let end = if i == num_chunks - 1 {
                x.len()
            } else {
                start + chunk_size
            };
            dot_product_packed_simd(&x[start..end], &y[start..end])
        })
        .collect();

    dot_products.par_iter().sum()
}

fn dot_product_rayon_simd(x: &[f64], y: &[f64]) -> f64 {
    let num_chunks = rayon::current_num_threads();
    let chunk_size = x.len() / num_chunks;

    let dot_products: Vec<f64> = (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            let start = i * chunk_size;
            let end = if i == num_chunks - 1 {
                x.len()
            } else {
                start + chunk_size
            };
            dot_prod_simd(&x[start..end], &y[start..end])
        })
        .collect();

    dot_products.par_iter().sum()
}

pub fn dot_prod_simd(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = a
        .array_chunks::<4>()
        .map(|&a| f64x4::from_array(a))
        .zip(b.array_chunks::<4>().map(|&b| f64x4::from_array(b)))
        .map(|(a, b)| a * b)
        .fold(f64x4::splat(0.0), std::ops::Add::add)
        .reduce_sum();

    let remain = a.len() - (a.len() % 4);
    sum += a[remain..]
        .iter()
        .zip(&b[remain..])
        .map(|(a, b)| a * b)
        .sum::<f64>();
    sum
}

fn main() {
    let num_threads: usize = num_cpus::get_physical();
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    println!("The number of threads is {:?}", num_threads);

    let mut timer: Timer = Timer {
        record_time: Instant::now(),
    };
    let normal = Normal::new(0f64, 3f64).unwrap();
    const N: usize = 2_100_000_000;
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
    let res: f64 = unsafe { ddot(n, &x, 1, &y, 1) };
    timer.toc("The time of Intel MKL cblas_ddot: ");
    println!("Result: {:?}", res);

    timer.tic();
    let res: f64 = x
        .iter()
        .zip(y.iter())
        .fold(0.0, |a, zipped| a + zipped.0 * zipped.1);
    timer.toc("The time of pure rust: ");
    println!("Result: {:?}", res);

    timer.tic();
    let res: f64 = x.par_iter().zip(y.par_iter()).map(|(a, b)| a * b).sum();
    timer.toc("The time of rayon: ");
    println!("Result: {:?}", res);

    timer.tic();
    let res: f64 = dot_product_packed_simd(&x, &y);
    timer.toc("The time of packed_simd: ");
    println!("Result: {:?}", res);

    timer.tic();
    let res: f64 = dot_prod_simd(&x, &y);
    timer.toc("The time of std::simd: ");
    println!("Result: {:?}", res);

    timer.tic();
    let res: f64 = dot_product_rayon_packed_simd(&x, &y);
    timer.toc("The time of rayon + packed_simd ddot: ");
    println!("Result: {:?}", res);

    timer.tic();
    let res: f64 = dot_product_rayon_simd(&x, &y);
    timer.toc("The time of rayon + std::simd: ");
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

    let mut r: Vec<f64> = vec![0f64; N];
    timer.tic();
    x.par_chunks(4)
        .map(packed_simd::f64x4::from_slice_unaligned)
        .zip(
            y.par_chunks(4)
                .map(packed_simd::f64x4::from_slice_unaligned),
        )
        .map(|(a, b)| a + b)
        .zip(r.par_chunks_mut(4))
        .for_each(|(v, slice)| {
            v.write_to_slice_unaligned(slice);
        });
    timer.toc("The time of rayon + SIMD vdAdd: ");
    println!("Result: {:?}, {:?}, {:?}", r[0], r[1], r[2]);

    // test on dgemv
    let (m, k) = (6000, 200000);
    let alpha: f64 = 2.0;
    let beta: f64 = 1.0;
    let mut a: Vec<f64> = vec![0f64; m * k];
    timer.tic();
    fill_vec_with_random_dist(&mut a, &normal);
    timer.toc(&format!(
        "The time of generating random normal number of a ({0}x{1}): ",
        m, k
    ));

    let mut b: Vec<f64> = vec![0f64; k];
    timer.tic();
    fill_vec_with_random_dist(&mut b, &normal);
    timer.toc(&format!(
        "The time of generating random normal number of a ({0}x1): ",
        k
    ));

    let mut c: Vec<f64> = vec![2f64; m];
    timer.tic();
    unsafe {
        dgemv(
            Layout::RowMajor,
            Transpose::None,
            m as i32,
            k as i32,
            alpha,
            &a,
            k as i32,
            &b,
            1i32,
            beta,
            &mut c,
            1i32,
        );
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

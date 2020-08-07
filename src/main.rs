use std::vec::Vec;

extern crate time;
use time::Instant;

extern crate rayon;
use rayon::prelude::*;

#[cfg(all(feature = "nightly", feature = "packed_simd"))]
extern crate packed_simd;
use packed_simd::f64x8;

extern crate cblas;
extern crate intel_mkl_src;
use cblas::ddot;

#[macro_use]
extern crate arrayref;

extern "C" {
    pub fn vdAdd(n: ::std::os::raw::c_int, a: *const f64, b: *const f64, r: *mut f64);
}

fn main() {
    const N: usize = 300_000_000;
    let n: i32 = N as i32;
    let x: Vec<f64> = vec![0.2; N];
    let y: Vec<f64> = vec![0.3; N];

    let start = Instant::now();
    let res = unsafe { ddot(n, &x, 1, &y, 1) };
    let end = Instant::now();
    println!("Intel MKL cblas_ddot: {:?}", res);
    println!("total {} milliseconds", (end - start).whole_microseconds() as f64 / 1000f64);

    let start = Instant::now();
    let res: f64 = x
        .par_iter()
        .zip(y.par_iter())
        .map(|(a, b)| a * b)
        .sum();
    let end = Instant::now();
    println!("rayon ddot: {:?}", res);
    println!("total {} milliseconds", (end - start).whole_microseconds() as f64 / 1000f64);

    let start = Instant::now();
    let res: f64 = x
        .par_chunks(8)
        .map(f64x8::from_slice_unaligned)
        .zip(y.par_chunks(8).map(f64x8::from_slice_unaligned))
        .map(|(a, b)| (a * b).sum())
        .sum();
    let end = Instant::now();
    println!("rayon & SIMD ddot: {:?}", res);
    println!("total {} milliseconds", (end - start).whole_microseconds() as f64 / 1000f64);

    let xa = array_ref!(x, 0, N);
    let ya = array_ref!(y, 0, N);
    let mut r: Vec<f64> = vec![0f64; N];
    let mut ra = array_mut_ref!(r, 0, N);
    let start = Instant::now();
    unsafe { vdAdd(n, xa.as_ptr(), ya.as_ptr(), ra.as_mut_ptr()) };
    let end = Instant::now();
    println!("Intel MKL vdAdd: {:?}, {:?}, {:?}", r[0], r[1], r[2]);
    println!("total {} milliseconds", (end - start).whole_nanoseconds() as f64 / 1000000f64);

    let start = Instant::now();
    let r: Vec<f64> = x
        .par_iter()
        .zip(y.par_iter())
        .map(|(a, b)| a + b)
        .collect();
    let end = Instant::now();
    println!("rayon vdAdd: {:?}, {:?}, {:?}", r[0], r[1], r[2]);
    println!("total {} milliseconds", (end - start).whole_nanoseconds() as f64 / 1000000f64);
    
    let mut r: Vec<f64> = vec![0f64; N];
    let start = Instant::now();
    x.par_chunks(8)
             .map(f64x8::from_slice_unaligned)
             .zip(y.par_chunks(8).map(f64x8::from_slice_unaligned))
             .map(|(a,b)| a+b)
             .zip(r.par_chunks_mut(8))
             .for_each(|(v, slice)| {
                v.write_to_slice_unaligned(slice);
             });
    let end = Instant::now();
    println!("rayon & SIMD vdAdd: {:?}, {:?}, {:?}", r[0], r[1], r[2]);
    println!("total {} milliseconds", (end - start).whole_nanoseconds() as f64 / 1000000f64);   
}

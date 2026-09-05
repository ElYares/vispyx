//! Rust backend for the vispyx binary morphology engine.
//!
//! This crate contains no validation and no policy. It receives arrays that
//! `vispyx.morphology_common` has already validated and normalized, and it
//! reproduces the reference Python engine bit for bit. Every error message
//! `vispyx` exposes is public contract and is raised on the Python side, so
//! the only errors raised here are internal-dispatch bugs.
//!
//! The three details that have to match exactly:
//!
//! 1. Padding is `np.pad(mode="reflect")`: the border is mirrored *without*
//!    being repeated, so `[1, 2, 3]` padded by one is `[2, 1, 2, 3, 2]`.
//! 2. Padding is recomputed on every iteration, exactly like the Python loop.
//!    Folding the iterations into a single wider window changes the borders.
//! 3. Input and output live in `{0, 1}`. The caller multiplies by 255.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Erode,
    Dilate,
}

/// Map a possibly out-of-range index onto `np.pad(mode="reflect")`.
///
/// `rem_euclid` first, so the fold below works for pad widths larger than the
/// axis itself: a 7x7 kernel over a 2-pixel axis mirrors several times.
#[inline]
fn reflect(index: isize, len: isize) -> usize {
    if len == 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let mut folded = index.rem_euclid(period);
    if folded >= len {
        folded = period - folded;
    }
    folded as usize
}

/// Run one erosion or dilation pass over `src`, writing into `dst`.
///
/// `offsets` holds the active kernel cells as `(dy, dx)` relative to the
/// center, so inactive cells cost nothing at all: the Python engine spends
/// them on a boolean mask, here they are simply absent.
fn sweep(
    src: &[u8],
    height: isize,
    width: isize,
    offsets: &[(isize, isize)],
    radius: (isize, isize),
    op: Op,
    dst: &mut [u8],
) {
    let (pad_y, pad_x) = radius;
    let stride = width as usize;

    for i in 0..height {
        // Rows whose whole window fits inside the array skip the reflection.
        let row_inside = i >= pad_y && i < height - pad_y;

        for j in 0..width {
            let inside = row_inside && j >= pad_x && j < width - pad_x;
            let mut value = if op == Op::Erode { 1u8 } else { 0u8 };

            for &(dy, dx) in offsets {
                let sample = if inside {
                    src[(i + dy) as usize * stride + (j + dx) as usize]
                } else {
                    src[reflect(i + dy, height) * stride + reflect(j + dx, width)]
                };

                match op {
                    Op::Erode => {
                        if sample == 0 {
                            value = 0;
                            break;
                        }
                    }
                    Op::Dilate => {
                        if sample != 0 {
                            value = 1;
                            break;
                        }
                    }
                }
            }

            dst[i as usize * stride + j as usize] = value;
        }
    }
}

/// Apply `iterations` binary erosions or dilations.
///
/// `image` and `kernel` are expected to hold only zeros and ones; the caller
/// guarantees it. `op` is `"erode"` or `"dilate"`.
#[pyfunction]
#[pyo3(signature = (image, kernel, iterations, op))]
fn binary_op<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, u8>,
    kernel: PyReadonlyArray2<'py, u8>,
    iterations: usize,
    op: &str,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let op = match op {
        "erode" => Op::Erode,
        "dilate" => Op::Dilate,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown native operation: {other}"
            )))
        }
    };

    let image = image.as_array();
    let kernel = kernel.as_array();

    let (height, width) = image.dim();
    let (kernel_height, kernel_width) = kernel.dim();
    let radius = ((kernel_height / 2) as isize, (kernel_width / 2) as isize);

    let mut offsets = Vec::with_capacity(kernel_height * kernel_width);
    for ky in 0..kernel_height {
        for kx in 0..kernel_width {
            if kernel[(ky, kx)] != 0 {
                offsets.push((ky as isize - radius.0, kx as isize - radius.1));
            }
        }
    }

    // `.iter()` walks in logical row-major order whatever the memory layout is,
    // so a sliced or transposed view lands here correctly.
    let mut current: Vec<u8> = image.iter().copied().collect();
    let mut next = vec![0u8; current.len()];

    for _ in 0..iterations {
        sweep(
            &current,
            height as isize,
            width as isize,
            &offsets,
            radius,
            op,
            &mut next,
        );
        std::mem::swap(&mut current, &mut next);
    }

    let result = numpy::ndarray::Array2::from_shape_vec((height, width), current)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Operations this build can handle. The Python side falls back for the rest.
#[pyfunction]
fn supported_ops() -> Vec<&'static str> {
    vec!["erode", "dilate"]
}

#[pymodule]
fn vispyx_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(binary_op, module)?)?;
    module.add_function(wrap_pyfunction!(supported_ops, module)?)?;
    Ok(())
}

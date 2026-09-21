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
//!
//! Zhang-Suen ([`zhang_suen`]) is the exception to point 1: it pads with zeros,
//! on purpose, exactly like `vpx_skeletonize`.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Erode,
    Dilate,
}

/// El nombre lo elige el despacho de Python, asi que un valor desconocido es un
/// bug interno y no una entrada del usuario.
fn parse_op(op: &str) -> PyResult<Op> {
    match op {
        "erode" => Ok(Op::Erode),
        "dilate" => Ok(Op::Dilate),
        other => Err(PyValueError::new_err(format!(
            "unknown native operation: {other}"
        ))),
    }
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

/// Active kernel cells as `(dy, dx)` offsets from the center, plus that center.
///
/// Las celdas inactivas desaparecen aqui: el motor de Python las paga en una
/// mascara booleana por ventana, y en Rust simplemente no estan.
fn active_offsets(
    kernel: &numpy::ndarray::ArrayView2<u8>,
) -> (Vec<(isize, isize)>, (isize, isize)) {
    let (height, width) = kernel.dim();
    let radius = ((height / 2) as isize, (width / 2) as isize);

    let mut offsets = Vec::with_capacity(height * width);
    for y in 0..height {
        for x in 0..width {
            if kernel[(y, x)] != 0 {
                offsets.push((y as isize - radius.0, x as isize - radius.1));
            }
        }
    }
    (offsets, radius)
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

/// Grayscale twin of [`sweep`]: `min` or `max` over the active support.
///
/// Genérica sobre enteros y no sobre cualquier numero a proposito. `Ord` es un
/// orden total; los flotantes solo tienen `PartialOrd`, y reproducir la
/// propagacion de `NaN` de `np.min` bit a bit no vale el riesgo. El lado Python
/// manda los flotantes a su propio bucle, que sigue siendo la referencia.
fn sweep_gray<T: Copy + Ord>(
    src: &[T],
    height: isize,
    width: isize,
    offsets: &[(isize, isize)],
    radius: (isize, isize),
    op: Op,
    dst: &mut [T],
) {
    let (pad_y, pad_x) = radius;
    let stride = width as usize;

    for i in 0..height {
        let row_inside = i >= pad_y && i < height - pad_y;

        for j in 0..width {
            let inside = row_inside && j >= pad_x && j < width - pad_x;

            // `validate_kernel` garantiza al menos una celda activa, asi que el
            // primer offset sirve de acumulador inicial y evita un Option por
            // pixel.
            let sample_at = |dy: isize, dx: isize| {
                if inside {
                    src[(i + dy) as usize * stride + (j + dx) as usize]
                } else {
                    src[reflect(i + dy, height) * stride + reflect(j + dx, width)]
                }
            };

            let (first_dy, first_dx) = offsets[0];
            let mut value = sample_at(first_dy, first_dx);

            for &(dy, dx) in &offsets[1..] {
                let sample = sample_at(dy, dx);
                value = match op {
                    Op::Erode => {
                        if sample < value {
                            sample
                        } else {
                            value
                        }
                    }
                    Op::Dilate => {
                        if sample > value {
                            sample
                        } else {
                            value
                        }
                    }
                };
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
    let op = parse_op(op)?;

    let image = image.as_array();
    let kernel = kernel.as_array();

    let (height, width) = image.dim();
    let (offsets, radius) = active_offsets(&kernel);

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

/// Cuerpo de `grayscale_op` una vez resuelto el dtype.
fn run_grayscale<'py, T>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, T>,
    kernel: PyReadonlyArray2<'py, u8>,
    iterations: usize,
    op: Op,
) -> PyResult<Bound<'py, PyAny>>
where
    T: numpy::Element + Copy + Ord,
{
    let image = image.as_array();
    let kernel = kernel.as_array();

    let (height, width) = image.dim();
    let (offsets, radius) = active_offsets(&kernel);

    let mut current: Vec<T> = image.iter().copied().collect();
    let mut next = current.clone();

    for _ in 0..iterations {
        sweep_gray(
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
    Ok(result.into_pyarray(py).into_any())
}

/// Apply `iterations` grayscale erosions (min) or dilations (max).
///
/// Despacha por dtype y devuelve el mismo tipo que recibio. Solo enteros: los
/// flotantes se quedan en el bucle de Python, ver [`sweep_gray`].
#[pyfunction]
#[pyo3(signature = (image, kernel, iterations, op))]
fn grayscale_op<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    kernel: PyReadonlyArray2<'py, u8>,
    iterations: usize,
    op: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let op = parse_op(op)?;

    macro_rules! dispatch {
        ($($dtype:ty),+ $(,)?) => {
            $(
                if let Ok(typed) = image.extract::<PyReadonlyArray2<$dtype>>() {
                    return run_grayscale::<$dtype>(py, typed, kernel, iterations, op);
                }
            )+
        };
    }

    dispatch!(u8, i8, u16, i16, u32, i32, u64, i64);

    Err(PyValueError::new_err(
        "unsupported dtype for the native grayscale engine",
    ))
}

/// One Zhang-Suen subpass over a zero-bordered buffer. Returns whether it deleted.
///
/// `grid` is `(height + 2) x (width + 2)` and its outer ring is always zero, so
/// the eight neighbors are read without bounds checks. That ring *is* the
/// padding: Zhang-Suen pads with zeros on purpose, unlike every other operation.
///
/// El borrado es diferido: primero se juntan los candidatos contra la imagen tal
/// como estaba al empezar la subpasada, y solo despues se apagan. Borrar en el
/// acto haria que cada decision dependiera del orden del recorrido.
fn zhang_suen_subpass(
    grid: &mut [u8],
    height: usize,
    width: usize,
    step: u8,
    to_remove: &mut Vec<usize>,
) -> bool {
    let stride = width + 2;
    to_remove.clear();

    for i in 1..=height {
        for j in 1..=width {
            let center = i * stride + j;
            if grid[center] != 1 {
                continue;
            }

            let p2 = grid[center - stride];
            let p3 = grid[center - stride + 1];
            let p4 = grid[center + 1];
            let p5 = grid[center + stride + 1];
            let p6 = grid[center + stride];
            let p7 = grid[center + stride - 1];
            let p8 = grid[center - 1];
            let p9 = grid[center - stride - 1];
            let ring = [p2, p3, p4, p5, p6, p7, p8, p9];

            let active: u8 = ring.iter().sum();
            if !(2..=6).contains(&active) {
                continue;
            }

            // Transiciones 0 -> 1 recorriendo p2..p9 y volviendo a p2.
            let mut transitions = 0;
            for k in 0..8 {
                if ring[k] == 0 && ring[(k + 1) % 8] == 1 {
                    transitions += 1;
                }
            }
            if transitions != 1 {
                continue;
            }

            let keep = if step == 0 {
                p2 * p4 * p6 != 0 || p4 * p6 * p8 != 0
            } else {
                p2 * p4 * p8 != 0 || p2 * p6 * p8 != 0
            };
            if keep {
                continue;
            }

            to_remove.push(center);
        }
    }

    for &index in to_remove.iter() {
        grid[index] = 0;
    }
    !to_remove.is_empty()
}

/// Zhang-Suen thinning until convergence, or until `max_iterations` full passes.
///
/// Una iteracion son las dos subpasadas, igual que en Python: el contador sube
/// aunque la primera haya borrado y la segunda no, y el corte por
/// `max_iterations` se evalua despues del de convergencia. El bucle vive aqui
/// adentro a proposito; cruzar la frontera con Python por iteracion costaria
/// una copia del arreglo cada vez.
#[pyfunction]
#[pyo3(signature = (image, max_iterations))]
fn zhang_suen<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, u8>,
    max_iterations: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let image = image.as_array();
    let (height, width) = image.dim();
    let stride = width + 2;

    let mut grid = vec![0u8; (height + 2) * stride];
    for ((i, j), &value) in image.indexed_iter() {
        grid[(i + 1) * stride + j + 1] = value;
    }

    let mut to_remove = Vec::new();
    let mut iterations = 0usize;

    loop {
        let mut changed = false;
        for step in 0..2u8 {
            if zhang_suen_subpass(&mut grid, height, width, step, &mut to_remove) {
                changed = true;
            }
        }

        iterations += 1;
        if !changed {
            break;
        }
        if let Some(limit) = max_iterations {
            if iterations >= limit {
                break;
            }
        }
    }

    let mut result = Vec::with_capacity(height * width);
    for i in 1..=height {
        result.extend_from_slice(&grid[i * stride + 1..i * stride + 1 + width]);
    }

    let result = numpy::ndarray::Array2::from_shape_vec((height, width), result)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(result.into_pyarray(py))
}

/// Binary operations this build can handle. The Python side falls back for the rest.
#[pyfunction]
fn supported_ops() -> Vec<&'static str> {
    vec!["erode", "dilate", "zhang_suen"]
}

/// Grayscale operations, and the dtypes the native engine accepts.
#[pyfunction]
fn supported_grayscale_dtypes() -> Vec<&'static str> {
    vec![
        "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64",
    ]
}

#[pymodule]
fn vispyx_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(binary_op, module)?)?;
    module.add_function(wrap_pyfunction!(grayscale_op, module)?)?;
    module.add_function(wrap_pyfunction!(zhang_suen, module)?)?;
    module.add_function(wrap_pyfunction!(supported_ops, module)?)?;
    module.add_function(wrap_pyfunction!(supported_grayscale_dtypes, module)?)?;
    Ok(())
}

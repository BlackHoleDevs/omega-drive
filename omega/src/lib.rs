#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyBytes, PyList};

use dashmap::DashMap;
use lazy_static::lazy_static;
use serde::{Serialize, Deserialize};

// ── OMEGA CORE TYPES ───────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct NeuralCell {
    pub spec_idx: i32,
    pub pool_idx: usize,
}

#[allow(dead_code)]
type OmegaIndex = DashMap<(String, String, String, String), NeuralCell>;

lazy_static! {
    static ref GLOBAL_INDEX: OmegaIndex = DashMap::new();
}

#[cfg(feature = "python")]
#[pyclass]
pub struct OmegaEngine {
    index: Arc<OmegaIndex>,
    table_index: Arc<DashMap<(String, String), Vec<(String, String)>>>,
    next_pool_idx: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "python")]
#[pymethods]
impl OmegaEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            index: Arc::new(DashMap::new()),
            table_index: Arc::new(DashMap::new()),
            next_pool_idx: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn get_batch_meta(&self, keys: Vec<(String, String, String, String)>) -> Vec<Option<(i32, usize)>> {
        keys.into_iter().map(|k| {
            self.index.get(&k).map(|cell| (cell.spec_idx, cell.pool_idx))
        }).collect()
    }

    pub fn get_cell_meta(&self, db: String, table: String, row: String, col: String) -> PyResult<Option<(i32, usize)>> {
        if let Some(cell) = self.index.get(&(db, table, row, col)) {
            Ok(Some((cell.spec_idx, cell.pool_idx)))
        } else {
            Ok(None)
        }
    }

    pub fn register_cell(&self, db: String, table: String, row: String, col: String, spec_idx: i32) -> usize {
        let pool_idx = self.next_pool_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.table_index.entry((db.clone(), table.clone()))
            .or_insert_with(Vec::new)
            .push((row.clone(), col.clone()));
        self.index.insert((db, table, row, col), NeuralCell { spec_idx, pool_idx });
        pool_idx
    }

    pub fn set(&self, db: String, table: String, row: String, col: String, spec_idx: i32, pool_idx: usize) {
        self.index.insert((db, table, row, col), NeuralCell { spec_idx, pool_idx });
        let current = self.next_pool_idx.load(std::sync::atomic::Ordering::SeqCst);
        if pool_idx >= current {
            self.next_pool_idx.store(pool_idx + 1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    pub fn register_batch(&self, db: String, table: String, row_ids: Vec<String>, col_names: Vec<String>, spec_indices: Vec<i32>) -> Vec<usize> {
        let count = spec_indices.len();
        let base_idx = self.next_pool_idx.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
        let mut results = Vec::with_capacity(count);
        let mut pairs = Vec::with_capacity(count);
        for i in 0..count {
            let pool_idx = base_idx + i;
            pairs.push((row_ids[i].clone(), col_names[i].clone()));
            self.index.insert(
                (db.clone(), table.clone(), row_ids[i].clone(), col_names[i].clone()),
                NeuralCell { spec_idx: spec_indices[i], pool_idx },
            );
            results.push(pool_idx);
        }
        self.table_index.entry((db, table))
            .or_insert_with(Vec::new)
            .extend(pairs);
        results
    }

    pub fn get_all_entries(&self) -> Vec<((String, String, String, String), (i32, usize))> {
        self.index.iter().map(|item| {
            let key = item.key().clone();
            let val = item.value();
            (key, (val.spec_idx, val.pool_idx))
        }).collect()
    }

    pub fn get_table_data(&self, py: Python<'_>, db_name: String, table_name: String) -> PyResult<(PyObject, Vec<i32>, Vec<usize>)> {
        let key = (db_name.clone(), table_name.clone());
        let pairs = match self.table_index.get(&key) {
            Some(p) => p.clone(),
            None => return Ok((PyBytes::new_bound(py, &[]).into_py(py).into_any(), vec![], vec![])),
        };

        let mut meta_blob = Vec::with_capacity(pairs.len() * 50);
        let mut p_indices = Vec::with_capacity(pairs.len());
        let mut s_indices = Vec::with_capacity(pairs.len());

        for (rid, col) in &pairs {
            if let Some(cell) = self.index.get(&(db_name.clone(), table_name.clone(), rid.clone(), col.clone())) {
                let v = cell.value();
                meta_blob.push(rid.len().min(255) as u8);
                meta_blob.extend_from_slice(rid.as_bytes());
                meta_blob.push(col.len().min(255) as u8);
                meta_blob.extend_from_slice(col.as_bytes());
                s_indices.push(v.spec_idx);
                p_indices.push(v.pool_idx);
            }
        }

        let meta_obj = PyBytes::new_bound(py, &meta_blob).into_py(py).into_any();
        Ok((meta_obj, s_indices, p_indices))
    }

    pub fn remove(&self, db: String, table: String, row: String, col: String) {
        self.index.remove(&(db.clone(), table.clone(), row.clone(), col.clone()));
        if let Some(mut pairs) = self.table_index.get_mut(&(db, table)) {
            pairs.retain(|(r, c)| r != &row || c != &col);
        }
    }

    pub fn clear(&self) {
        self.index.clear();
        self.table_index.clear();
        self.next_pool_idx.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }
}

#[cfg(feature = "python")]
#[pyfunction]
fn pack_cells_to_bytes(py: Python<'_>, cells: &Bound<'_, PyList>, max_len: usize) -> PyResult<PyObject> {
    let mut buffer = Vec::with_capacity(cells.len() * max_len);
    for item in cells.iter() {
        let bytes = if let Ok(b) = item.extract::<Vec<u8>>() { b }
                    else if let Ok(s) = item.extract::<String>() { s.into_bytes() }
                    else if let Ok(val) = item.getattr("value") {
                         if let Ok(b) = val.extract::<Vec<u8>>() { b }
                         else { val.extract::<String>()?.into_bytes() }
                    } else { item.to_string().into_bytes() };
        let len = bytes.len().min(max_len - 5);
        buffer.push(1);
        buffer.extend_from_slice(&(len as u32).to_be_bytes());
        buffer.extend_from_slice(&bytes[..len]);
        let padding = max_len - 5 - len;
        if padding > 0 { buffer.extend(std::iter::repeat(0).take(padding)); }
    }
    Ok(PyBytes::new_bound(py, &buffer).into_py(py).into_any())
}

#[cfg(feature = "python")]
#[pyfunction]
fn batch_decode_cells(py: Python<'_>, buffer: &[u8], cell_size: usize) -> PyResult<Vec<PyObject>> {
    let num_cells = buffer.len() / cell_size;
    let mut results = Vec::with_capacity(num_cells);
    for i in 0..num_cells {
        let start = i * cell_size;
        let cell_data = &buffer[start..start + cell_size];
        if cell_data.len() < 5 { results.push(py.None()); continue; }
        let type_tag = cell_data[0];
        let data_len = u32::from_be_bytes(cell_data[1..5].try_into().unwrap_or([0; 4])) as usize;
        let actual_data = &cell_data[5..(5 + data_len).min(cell_size)];
        match type_tag {
            1 => results.push(String::from_utf8_lossy(actual_data).into_owned().into_py(py)),
            _ => results.push(PyBytes::new_bound(py, actual_data).into_py(py).into_any()),
        }
    }
    Ok(results)
}

#[cfg(feature = "python")]
#[pyfunction]
fn batch_decode_to_blob(py: Python<'_>, buffer: &[u8], cell_size: usize) -> PyResult<PyObject> {
    let num_cells = buffer.len() / cell_size;
    let mut out = Vec::with_capacity(buffer.len());
    for i in 0..num_cells {
        let start = i * cell_size;
        let cell_data = &buffer[start..start + cell_size];
        if cell_data.len() < 5 { out.extend_from_slice(&0u32.to_le_bytes()); continue; }
        let data_len = u32::from_be_bytes(cell_data[1..5].try_into().unwrap_or([0; 4])) as usize;
        let actual_data = &cell_data[5..(5 + data_len).min(cell_size)];
        out.extend_from_slice(&(actual_data.len() as u32).to_le_bytes());
        out.extend_from_slice(actual_data);
    }
    Ok(PyBytes::new_bound(py, &out).into_py(py).into_any())
}

#[cfg(feature = "python")]
#[pyfunction]
fn pack_columnar_data_blobs(py: Python<'_>, meta_blob: &[u8], values_blob: &[u8]) -> PyResult<PyObject> {
    let mut buffer = Vec::with_capacity(meta_blob.len() + values_blob.len() + 1024);
    buffer.extend_from_slice(b"ANS1");
    let mut count = 0; let mut m_pos = 0;
    while m_pos < meta_blob.len() {
        let r_len = meta_blob[m_pos] as usize; m_pos += 1 + r_len;
        if m_pos >= meta_blob.len() { break; }
        let c_len = meta_blob[m_pos] as usize; m_pos += 1 + c_len; count += 1;
    }
    buffer.extend_from_slice(&(count as u32).to_le_bytes());
    let mut m_pos = 0; let mut v_pos = 0;
    for _ in 0..count {
        let r_len = meta_blob[m_pos] as usize; let rid = &meta_blob[m_pos+1 .. m_pos+1+r_len]; m_pos += 1 + r_len;
        let c_len = meta_blob[m_pos] as usize; let col = &meta_blob[m_pos+1 .. m_pos+1+c_len]; m_pos += 1 + c_len;
        let v_len = u32::from_le_bytes(values_blob[v_pos..v_pos+4].try_into().unwrap()) as usize;
        let val = &values_blob[v_pos+4 .. v_pos+4+v_len]; v_pos += 4 + v_len;
        buffer.push(r_len as u8); buffer.extend_from_slice(rid);
        buffer.push(c_len as u8); buffer.extend_from_slice(col);
        buffer.extend_from_slice(&(v_len as u32).to_le_bytes()); buffer.extend_from_slice(val);
    }
    Ok(PyBytes::new_bound(py, &buffer).into_py(py).into_any())
}

#[cfg(feature = "python")]
#[pymodule]
fn airdb_core_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OmegaEngine>()?;
    m.add_function(wrap_pyfunction!(pack_cells_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(batch_decode_cells, m)?)?;
    m.add_function(wrap_pyfunction!(batch_decode_to_blob, m)?)?;
    m.add_function(wrap_pyfunction!(pack_columnar_data_blobs, m)?)?;
    Ok(())
}

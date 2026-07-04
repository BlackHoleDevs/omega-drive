#![allow(dead_code)]
use std::str;

pub enum Command<'a> {
    Ping,
    Set(&'a str, &'a [u8]),
    Get(&'a str),
    Del(&'a str),
    Exists(&'a str),
    MSetHeader(usize), // num_args
    MGetHeader(usize), // num_args
    Select(u8),
    Save,
    BgSave,
    FlushDb,
    Info,
    HmSet(&'a str, Vec<(&'a str, &'a [u8])>),
    HGetAll(&'a str),
    ZAdd(&'a str),
    VAdd(&'a str, Vec<f32>),
    VSearch(usize, Vec<f32>),
    Unknown(&'a str),
}

/// A super-fast, zero-allocation RESP parser for specific commands.
pub fn parse_command_fast(buf: &[u8]) -> Option<(Command<'_>, usize)> {
    if buf.is_empty() || buf[0] != b'*' { return None; }
    
    let mut pos = 1;
    let (num_args, len) = parse_int(buf, pos)?;
    pos += len;
    if pos + 2 > buf.len() || &buf[pos..pos+2] != b"\r\n" { return None; }
    pos += 2;
    
    if num_args == 0 { return None; }
    
    // Parse the first argument (Command Name)
    if pos >= buf.len() || buf[pos] != b'$' { return None; }
    pos += 1;
    let (arg_len, len) = parse_int(buf, pos)?;
    pos += len;
    if pos + 2 > buf.len() || &buf[pos..pos+2] != b"\r\n" { return None; }
    pos += 2;
    
    if pos + arg_len as usize > buf.len() { return None; }
    let cmd_raw = &buf[pos .. pos + arg_len as usize];
    pos += arg_len as usize;
    if pos + 2 > buf.len() || &buf[pos..pos+2] != b"\r\n" { return None; }
    pos += 2;

    let cmd_name = str::from_utf8(cmd_raw).ok()?;
    let cmd = match cmd_name.to_uppercase().as_str() {
        "PING" => Command::Ping,
        "SET" => {
            if num_args < 3 { return None; }
            let (key, len) = parse_bulk_str(buf, pos)?;
            pos += len;
            let (val, len) = parse_bulk_str(buf, pos)?;
            pos += len;
            Command::Set(str::from_utf8(key).ok()?, val)
        }
        "GET" => {
            if num_args < 2 { return None; }
            let (key, len) = parse_bulk_str(buf, pos)?;
            pos += len;
            Command::Get(str::from_utf8(key).ok()?)
        }
        "DEL" => {
            if num_args < 2 { return None; }
            let (key, len) = parse_bulk_str(buf, pos)?;
            pos += len;
            Command::Del(str::from_utf8(key).ok()?)
        }
        "MGET" => Command::MGetHeader(num_args as usize - 1),
        "MSET" => Command::MSetHeader(num_args as usize - 1),
        "SELECT" => {
            if num_args < 2 { return None; }
            let (val, len) = parse_bulk_str(buf, pos)?;
            pos += len;
            let db_idx = str::from_utf8(val).ok()?.parse::<u8>().ok()?;
            Command::Select(db_idx)
        }
        "SAVE" => Command::Save,
        "BGSAVE" => Command::BgSave,
        "FLUSHDB" => Command::FlushDb,
        "INFO" => Command::Info,
        "HMSET" => {
            if num_args < 4 { return None; }
            let (key, len) = parse_bulk_str(buf, pos)?;
            pos += len;
            let num_pairs = (num_args as usize - 2) / 2;
            let mut fields = Vec::with_capacity(num_pairs);
            let mut current_pos = pos;
            for _ in 0..num_pairs {
                let (f_key, len) = parse_bulk_str(buf, current_pos)?;
                current_pos += len;
                let (f_val, len) = parse_bulk_str(buf, current_pos)?;
                current_pos += len;
                fields.push((str::from_utf8(f_key).ok()?, f_val));
            }
            pos = current_pos;
            Command::HmSet(str::from_utf8(key).ok()?, fields)
        }
        "HGETALL" => {
            if num_args < 2 { return None; }
            let (key, len) = parse_bulk_str(buf, pos)?;
            pos += len;
            Command::HGetAll(str::from_utf8(key).ok()?)
        }
        "ZADD" => {
            if num_args < 4 { return None; }
            let (key, len) = parse_bulk_str(buf, pos)?;
            pos += len;
            Command::ZAdd(str::from_utf8(key).ok()?)
        }
        "VADD" => {
            if num_args < 3 { return None; }
            let (key, len) = parse_bulk_str(buf, pos)?;
            pos += len;
            let num_floats = num_args as usize - 2;
            let mut floats = Vec::with_capacity(num_floats);
            let mut current_pos = pos;
            for _ in 0..num_floats {
                let (f_bytes, len) = parse_bulk_str(buf, current_pos)?;
                current_pos += len;
                let f_str = str::from_utf8(f_bytes).ok()?;
                let f_val = f_str.parse::<f32>().ok()?;
                floats.push(f_val);
            }
            pos = current_pos;
            Command::VAdd(str::from_utf8(key).ok()?, floats)
        }
        "VSEARCH" => {
            if num_args < 3 { return None; }
            let (k_bytes, len) = parse_bulk_str(buf, pos)?;
            pos += len;
            let k_str = str::from_utf8(k_bytes).ok()?;
            let k = k_str.parse::<usize>().ok()?;
            let num_floats = num_args as usize - 2;
            let mut floats = Vec::with_capacity(num_floats);
            let mut current_pos = pos;
            for _ in 0..num_floats {
                let (f_bytes, len) = parse_bulk_str(buf, current_pos)?;
                current_pos += len;
                let f_str = str::from_utf8(f_bytes).ok()?;
                let f_val = f_str.parse::<f32>().ok()?;
                floats.push(f_val);
            }
            pos = current_pos;
            Command::VSearch(k, floats)
        }
        _ => Command::Unknown(cmd_name),
    };
    
    Some((cmd, pos))
}

pub fn parse_bulk_str(buf: &[u8], mut pos: usize) -> Option<(&[u8], usize)> {
    let start = pos;
    if pos >= buf.len() || buf[pos] != b'$' { return None; }
    pos += 1;
    let (arg_len, len) = parse_int(buf, pos)?;
    pos += len;
    if pos + 2 > buf.len() || &buf[pos..pos+2] != b"\r\n" { return None; }
    pos += 2;
    
    if pos + arg_len as usize > buf.len() { return None; }
    let data = &buf[pos .. pos + arg_len as usize];
    pos += arg_len as usize;
    if pos + 2 > buf.len() || &buf[pos..pos+2] != b"\r\n" { return None; }
    pos += 2;
    
    Some((data, pos - start))
}

fn parse_int(buf: &[u8], start: usize) -> Option<(u32, usize)> {
    let mut i = start;
    let mut val = 0u32;
    while i < buf.len() && buf[i].is_ascii_digit() {
        val = val * 10 + (buf[i] - b'0') as u32;
        i += 1;
    }
    if i == start { None } else { Some((val, i - start)) }
}

pub fn make_error(msg: &str) -> Vec<u8> {
    let mut resp = Vec::with_capacity(msg.len() + 3);
    resp.push(b'-');
    resp.extend_from_slice(msg.as_bytes());
    resp.extend_from_slice(b"\r\n");
    resp
}

pub fn make_ok() -> &'static [u8] {
    b"+OK\r\n"
}

pub fn make_bulk_string(data: &[u8]) -> Vec<u8> {
    let mut resp = Vec::with_capacity(data.len() + 16);
    resp.push(b'$');
    resp.extend_from_slice(data.len().to_string().as_bytes());
    resp.extend_from_slice(b"\r\n");
    resp.extend_from_slice(data);
    resp.extend_from_slice(b"\r\n");
    resp
}

pub fn make_null() -> &'static [u8] {
    b"$-1\r\n"
}

pub fn make_pong() -> &'static [u8] {
    b"+PONG\r\n"
}

pub fn make_int(val: i64) -> Vec<u8> {
    let mut resp = Vec::with_capacity(16);
    resp.push(b':');
    resp.extend_from_slice(val.to_string().as_bytes());
    resp.extend_from_slice(b"\r\n");
    resp
}

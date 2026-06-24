use bytes::BytesMut;

pub struct FlatHash;

impl FlatHash {
    /// Serializes key-value pairs into a contiguous packed binary buffer
    pub fn serialize(fields: &[(&str, &[u8])]) -> Vec<u8> {
        let num_fields = fields.len() as u32;
        let mut capacity = 4;
        for (k, v) in fields {
            capacity += 4 + 4 + k.len() + v.len();
        }
        let mut buf = Vec::with_capacity(capacity);
        buf.extend_from_slice(&num_fields.to_be_bytes());
        for (k, v) in fields {
            let k_len = k.len() as u32;
            let v_len = v.len() as u32;
            buf.extend_from_slice(&k_len.to_be_bytes());
            buf.extend_from_slice(&v_len.to_be_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(v);
        }
        buf
    }

    /// Merges existing packed binary buffer with new key-value pairs
    pub fn merge(old_buf: &[u8], new_fields: &[(&str, &[u8])]) -> Vec<u8> {
        if old_buf.len() < 4 {
            return Self::serialize(new_fields);
        }
        
        let mut fields_map = std::collections::HashMap::new();
        
        // Parse old fields
        let num_old_fields = u32::from_be_bytes([old_buf[0], old_buf[1], old_buf[2], old_buf[3]]) as usize;
        let mut pos = 4;
        for _ in 0..num_old_fields {
            if pos + 8 > old_buf.len() { break; }
            let k_len = u32::from_be_bytes([old_buf[pos], old_buf[pos+1], old_buf[pos+2], old_buf[pos+3]]) as usize;
            let v_len = u32::from_be_bytes([old_buf[pos+4], old_buf[pos+5], old_buf[pos+6], old_buf[pos+7]]) as usize;
            pos += 8;
            if pos + k_len + v_len > old_buf.len() { break; }
            let k = &old_buf[pos..pos+k_len];
            let v = &old_buf[pos+k_len..pos+k_len+v_len];
            pos += k_len + v_len;
            fields_map.insert(k.to_vec(), v.to_vec());
        }
        
        // Insert/overwrite new fields
        for (k, v) in new_fields {
            fields_map.insert(k.as_bytes().to_vec(), v.to_vec());
        }
        
        // Serialize back
        let num_fields = fields_map.len() as u32;
        let mut capacity = 4;
        for (k, v) in &fields_map {
            capacity += 4 + 4 + k.len() + v.len();
        }
        let mut buf = Vec::with_capacity(capacity);
        buf.extend_from_slice(&num_fields.to_be_bytes());
        for (k, v) in fields_map {
            let k_len = k.len() as u32;
            let v_len = v.len() as u32;
            buf.extend_from_slice(&k_len.to_be_bytes());
            buf.extend_from_slice(&v_len.to_be_bytes());
            buf.extend_from_slice(&k);
            buf.extend_from_slice(&v);
        }
        buf
    }

    /// Deserializes packed binary buffer directly into a RESP Array reply
    pub fn deserialize_into_resp(buf: &[u8], write_buf: &mut BytesMut) {
        if buf.len() < 4 {
            write_buf.extend_from_slice(b"*0\r\n");
            return;
        }
        let num_fields = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        write_buf.extend_from_slice(format!("*{}\r\n", num_fields * 2).as_bytes());
        let mut pos = 4;
        for _ in 0..num_fields {
            if pos + 8 > buf.len() { break; }
            let k_len = u32::from_be_bytes([buf[pos], buf[pos+1], buf[pos+2], buf[pos+3]]) as usize;
            let v_len = u32::from_be_bytes([buf[pos+4], buf[pos+5], buf[pos+6], buf[pos+7]]) as usize;
            pos += 8;
            if pos + k_len + v_len > buf.len() { break; }
            let k = &buf[pos..pos+k_len];
            let v = &buf[pos+k_len..pos+k_len+v_len];
            pos += k_len + v_len;

            // Stream key
            write_buf.extend_from_slice(format!("${}\r\n", k_len).as_bytes());
            write_buf.extend_from_slice(k);
            write_buf.extend_from_slice(b"\r\n");

            // Stream value
            write_buf.extend_from_slice(format!("${}\r\n", v_len).as_bytes());
            write_buf.extend_from_slice(v);
            write_buf.extend_from_slice(b"\r\n");
        }
    }
}

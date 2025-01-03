use crate::{Error, Result};

pub(crate) fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn put_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn put_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    let len = u32::try_from(value.len())
        .map_err(|_| Error::InvalidInput("value exceeds 4 GiB storage limit".to_owned()))?;
    put_u32(out, len);
    out.extend_from_slice(value);
    Ok(())
}

pub(crate) fn read_u32(input: &mut &[u8]) -> Result<u32> {
    let bytes = take(input, 4)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

pub(crate) fn read_u64(input: &mut &[u8]) -> Result<u64> {
    let bytes = take(input, 8)?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

pub(crate) fn read_i64(input: &mut &[u8]) -> Result<i64> {
    let bytes = take(input, 8)?;
    Ok(i64::from_le_bytes(bytes.try_into().unwrap()))
}

pub(crate) fn read_bytes<'a>(input: &mut &'a [u8]) -> Result<&'a [u8]> {
    let len = read_u32(input)? as usize;
    take(input, len)
}

fn take<'a>(input: &mut &'a [u8], len: usize) -> Result<&'a [u8]> {
    if input.len() < len {
        return Err(Error::Corrupt("truncated record"));
    }
    let (head, tail) = input.split_at(len);
    *input = tail;
    Ok(head)
}

pub(crate) fn checksum(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        let index = ((crc as u8) ^ byte) as usize;
        crc = CRC32C_TABLE[index] ^ (crc >> 8);
    }
    !crc
}

const CRC32C_TABLE: [u32; 256] = crc32c_table();

const fn crc32c_table() -> [u32; 256] {
    let mut table = [0_u32; 256];
    let mut index = 0;
    while index < table.len() {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 == 1 {
                0x82f63b78 ^ (value >> 1)
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_round_trip() {
        let mut bytes = Vec::new();
        put_u32(&mut bytes, 42);
        put_i64(&mut bytes, -91);
        put_bytes(&mut bytes, b"hello").unwrap();
        let mut input = bytes.as_slice();
        assert_eq!(read_u32(&mut input).unwrap(), 42);
        assert_eq!(read_i64(&mut input).unwrap(), -91);
        assert_eq!(read_bytes(&mut input).unwrap(), b"hello");
        assert!(input.is_empty());
    }

    #[test]
    fn crc32c_matches_standard_vector() {
        assert_eq!(checksum(b"123456789"), 0xe306_9283);
    }
}

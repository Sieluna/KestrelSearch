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

pub(crate) fn put_var_u32(out: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

pub(crate) fn read_var_u32(input: &mut &[u8]) -> Result<u32> {
    let mut value = 0_u32;
    for shift in [0, 7, 14, 21, 28] {
        let (&byte, rest) = input
            .split_first()
            .ok_or(Error::Corrupt("truncated variable integer"))?;
        *input = rest;
        if shift == 28 && byte > 0x0f {
            return Err(Error::Corrupt("variable integer overflow"));
        }
        value |= u32::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(Error::Corrupt("variable integer overflow"))
}

pub(crate) fn put_packed_u32s(out: &mut Vec<u8>, values: &[u32]) {
    for block in values.chunks(128) {
        put_packed_block(out, block);
    }
}

pub(crate) fn put_packed_deltas(out: &mut Vec<u8>, values: &[u32]) {
    let mut previous = 0_u32;
    let mut first = true;
    let mut deltas = [0_u32; 128];
    for block in values.chunks(128) {
        for (index, value) in block.iter().copied().enumerate() {
            deltas[index] = if first {
                first = false;
                value
            } else {
                value - previous
            };
            previous = value;
        }
        put_packed_block(out, &deltas[..block.len()]);
    }
}

fn put_packed_block(out: &mut Vec<u8>, values: &[u32]) {
    debug_assert!(!values.is_empty() && values.len() <= 128);
    out.push(values.len() as u8);
    let width = values
        .iter()
        .copied()
        .max()
        .map_or(0, |value| (u32::BITS - value.leading_zeros()) as u8);
    out.push(width);
    let mut buffer = 0_u64;
    let mut buffered = 0_u32;
    for value in values {
        buffer |= u64::from(*value) << buffered;
        buffered += u32::from(width);
        while buffered >= 8 {
            out.push(buffer as u8);
            buffer >>= 8;
            buffered -= 8;
        }
    }
    if buffered != 0 {
        out.push(buffer as u8);
    }
}

pub(crate) fn read_packed_u32s(mut input: &[u8], expected: usize) -> Result<Vec<u32>> {
    let mut values = Vec::with_capacity(expected);
    while values.len() < expected {
        let (&count, rest) = input
            .split_first()
            .ok_or(Error::Corrupt("truncated packed block"))?;
        let (&width, rest) = rest
            .split_first()
            .ok_or(Error::Corrupt("truncated packed block"))?;
        input = rest;
        let count = count as usize;
        if count == 0 || count > 128 || count > expected - values.len() || width > 32 {
            return Err(Error::Corrupt("invalid packed block header"));
        }
        let byte_len = (count * usize::from(width)).div_ceil(8);
        if input.len() < byte_len {
            return Err(Error::Corrupt("truncated packed block payload"));
        }
        let (encoded, rest) = input.split_at(byte_len);
        input = rest;
        let mut byte_index = 0_usize;
        let mut buffer = 0_u64;
        let mut buffered = 0_u32;
        let mask = if width == 32 {
            u64::from(u32::MAX)
        } else if width == 0 {
            0
        } else {
            (1_u64 << width) - 1
        };
        for _ in 0..count {
            while buffered < u32::from(width) {
                buffer |= u64::from(encoded[byte_index]) << buffered;
                byte_index += 1;
                buffered += 8;
            }
            values.push((buffer & mask) as u32);
            buffer >>= width;
            buffered -= u32::from(width);
        }
    }
    if !input.is_empty() {
        return Err(Error::Corrupt("trailing packed integers"));
    }
    Ok(values)
}

pub(crate) fn read_packed_deltas(input: &[u8], expected: usize) -> Result<Vec<u32>> {
    let mut values = read_packed_u32s(input, expected)?;
    let mut previous = 0_u32;
    for (index, value) in values.iter_mut().enumerate() {
        if index != 0 && *value == 0 {
            return Err(Error::Corrupt("zero document delta"));
        }
        previous = if index == 0 {
            *value
        } else {
            previous
                .checked_add(*value)
                .ok_or(Error::Corrupt("document delta overflow"))?
        };
        *value = previous;
    }
    Ok(values)
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
    fn packed_blocks_and_deltas_round_trip() {
        let values: Vec<_> = (0..513_u32)
            .map(|index| index.wrapping_mul(index).rotate_left(index % 31))
            .collect();
        let mut packed = Vec::new();
        put_packed_u32s(&mut packed, &values);
        assert_eq!(read_packed_u32s(&packed, values.len()).unwrap(), values);

        let increasing: Vec<_> = (0..513_u32)
            .scan(0_u32, |current, step| {
                *current += step % 17 + 1;
                Some(*current)
            })
            .collect();
        packed.clear();
        put_packed_deltas(&mut packed, &increasing);
        assert_eq!(
            read_packed_deltas(&packed, increasing.len()).unwrap(),
            increasing
        );
    }

    #[test]
    fn variable_integers_cover_u32_domain() {
        let values = [0, 1, 127, 128, 16_383, 16_384, u32::MAX];
        let mut encoded = Vec::new();
        for value in values {
            put_var_u32(&mut encoded, value);
        }
        let mut input = encoded.as_slice();
        for value in values {
            assert_eq!(read_var_u32(&mut input).unwrap(), value);
        }
        assert!(input.is_empty());
    }

    #[test]
    fn crc32c_matches_standard_vector() {
        assert_eq!(checksum(b"123456789"), 0xe306_9283);
    }
}

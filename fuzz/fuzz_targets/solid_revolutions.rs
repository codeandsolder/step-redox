#![no_main]

use libfuzzer_sys::fuzz_target;
use step_redox::detect_solid_revolutions_bytes;

fuzz_target!(|data: &[u8]| {
    let _ = detect_solid_revolutions_bytes(data);
});

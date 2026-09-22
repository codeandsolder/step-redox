#![no_main]

use libfuzzer_sys::fuzz_target;
use step_redox::{Options, clean_bytes};

fuzz_target!(|data: &[u8]| {
    let Ok(first) = clean_bytes(data, &Options::default()) else {
        return;
    };

    let second = clean_bytes(&first.bytes, &Options::default())
        .expect("step-redox output must remain parseable by step-redox");
    assert_eq!(
        first.bytes, second.bytes,
        "successful safe cleanup must be byte-idempotent"
    );
});

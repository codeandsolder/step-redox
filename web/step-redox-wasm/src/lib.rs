use serde_json::json;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct StepResult {
    bytes: Vec<u8>,
    report_json: String,
}

#[wasm_bindgen]
impl StepResult {
    #[wasm_bindgen(getter)]
    pub fn bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    #[wasm_bindgen(getter, js_name = reportJson)]
    pub fn report_json(&self) -> String {
        self.report_json.clone()
    }
}

fn profile(name: &str) -> Result<step_redox::OutputProfile, JsValue> {
    match name {
        "compat" => Ok(step_redox::OutputProfile::Compat),
        "compact" => Ok(step_redox::OutputProfile::Compact),
        other => Err(JsValue::from_str(&format!(
            "unknown profile {other:?}; expected \"compat\" or \"compact\""
        ))),
    }
}

fn anchor(name: &str) -> Result<step_redox::patterns::PatternAnchor, JsValue> {
    match name {
        "start" => Ok(step_redox::patterns::PatternAnchor::Start),
        "center" => Ok(step_redox::patterns::PatternAnchor::Center),
        "end" => Ok(step_redox::patterns::PatternAnchor::End),
        other => Err(JsValue::from_str(&format!(
            "unknown anchor {other:?}; expected \"start\", \"center\", or \"end\""
        ))),
    }
}

#[wasm_bindgen]
pub fn optimize_step(input: &[u8], profile_name: &str) -> Result<StepResult, JsValue> {
    let profile = profile(profile_name)?;
    let cleaned = step_redox::clean_bytes(input, &step_redox::Options::for_profile(profile))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let report_json = serde_json::to_string(&json!({
        "stats": cleaned.stats,
        "compatibility": cleaned.compatibility,
        "patterns": cleaned.patterns,
        "periodic_bodies": cleaned.periodic_bodies,
    }))
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(StepResult {
        bytes: cleaned.bytes,
        report_json,
    })
}

#[wasm_bindgen]
pub fn resize_linear_pattern(
    input: &[u8],
    pattern_index: usize,
    new_count: usize,
    anchor_name: &str,
) -> Result<StepResult, JsValue> {
    let edited = step_redox::resize_linear_pattern_bytes(
        input,
        pattern_index,
        new_count,
        anchor(anchor_name)?,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let report_json = serde_json::to_string(&json!({
        "resize": edited.resize,
        "compatibility": edited.compatibility,
        "patterns": edited.patterns,
        "periodic_bodies": edited.periodic_bodies,
    }))
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(StepResult {
        bytes: edited.bytes,
        report_json,
    })
}

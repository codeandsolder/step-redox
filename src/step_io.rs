use anyhow::{Context, Result, bail};
use ruststep::ast::{EntityInstance, Exchange, Name, Parameter, Record};
use std::fmt::Write as _;

#[derive(Debug)]
pub(crate) struct ParsedExchange {
    pub exchange: Exchange,
    pub input_encoding: &'static str,
}

impl ParsedExchange {
    pub(crate) fn parse(input: &[u8]) -> Result<Self> {
        let (input_text, input_encoding) = decode_input(input)?;
        let exchange =
            ruststep::parser::parse(&input_text).context("parse STEP exchange structure")?;
        require_supported_sections(&exchange)?;
        Ok(Self {
            exchange,
            input_encoding,
        })
    }
}

pub(crate) fn require_supported_sections(exchange: &Exchange) -> Result<()> {
    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("ANCHOR/REFERENCE/SIGNATURE sections are not yet supported");
    }
    Ok(())
}

///
/// # Errors
/// Returns an error if the exchange structure contains data that cannot be serialized as supported STEP text.
pub fn write_exchange(exchange: &Exchange) -> Result<String> {
    require_supported_sections(exchange)?;

    let mut out = String::with_capacity(
        exchange
            .data
            .iter()
            .map(|d| d.entities.len())
            .sum::<usize>()
            * 48,
    );
    out.push_str("ISO-10303-21;\nHEADER;\n");
    for record in &exchange.header {
        write_record(record, &mut out);
        out.push_str(";\n");
    }
    out.push_str("ENDSEC;\n");

    for section in &exchange.data {
        if section.meta.is_empty() {
            out.push_str("DATA;\n");
        } else {
            out.push_str("DATA(");
            for (i, param) in section.meta.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_param(param, &mut out);
            }
            out.push_str(");\n");
        }
        for entity in &section.entities {
            write_entity(entity, &mut out);
            out.push('\n');
        }
        out.push_str("ENDSEC;\n");
    }
    out.push_str("END-ISO-10303-21;\n");
    Ok(out)
}

fn write_entity(entity: &EntityInstance, out: &mut String) {
    match entity {
        EntityInstance::Simple { id, record } => {
            let _ = write!(out, "#{id}=");
            write_record(record, out);
            out.push(';');
        }
        EntityInstance::Complex { id, subsuper } => {
            let _ = write!(out, "#{id}=(");
            for record in &subsuper.0 {
                write_record(record, out);
            }
            out.push_str(");");
        }
    }
}

fn write_record(record: &Record, out: &mut String) {
    out.push_str(&record.name);
    write_param(&record.parameter, out);
}

fn write_param(param: &Parameter, out: &mut String) {
    match param {
        Parameter::Typed { keyword, parameter } => {
            out.push_str(keyword);
            out.push('(');
            write_param(parameter, out);
            out.push(')');
        }
        Parameter::Integer(v) => {
            let _ = write!(out, "{v}");
        }
        Parameter::Real(v) => out.push_str(&format_real(*v)),
        Parameter::String(s) => write_step_string(s, out),
        Parameter::Enumeration(s) => {
            out.push('.');
            out.push_str(s);
            out.push('.');
        }
        Parameter::List(items) => {
            out.push('(');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_param(item, out);
            }
            out.push(')');
        }
        Parameter::Ref(Name::Entity(id)) => {
            let _ = write!(out, "#{id}");
        }
        Parameter::Ref(Name::Value(id)) => {
            let _ = write!(out, "@{id}");
        }
        Parameter::Ref(Name::ConstantEntity(s)) => {
            out.push('#');
            out.push_str(s);
        }
        Parameter::Ref(Name::ConstantValue(s)) => {
            out.push('@');
            out.push_str(s);
        }
        Parameter::NotProvided => out.push('$'),
        Parameter::Omitted => out.push('*'),
    }
}

pub fn decode_input(input: &[u8]) -> Result<(std::borrow::Cow<'_, str>, &'static str)> {
    if let Ok(s) = std::str::from_utf8(input) {
        return Ok((std::borrow::Cow::Borrowed(s), "utf-8"));
    }

    let (decoded, _used_encoding, had_errors) = encoding_rs::GBK.decode(input);
    if had_errors {
        bail!("STEP input is neither valid UTF-8 nor valid GBK");
    }
    Ok((decoded, "gbk"))
}

pub(super) fn write_step_string(s: &str, out: &mut String) {
    out.push('\'');

    let flush_non_ascii = |buf: &mut String, out: &mut String| {
        if buf.is_empty() {
            return;
        }
        out.push_str("\\X2\\");
        for unit in buf.encode_utf16() {
            let _ = write!(out, "{unit:04X}");
        }
        out.push_str("\\X0\\");
        buf.clear();
    };

    let mut non_ascii = String::new();
    for ch in s.chars() {
        if ch.is_ascii() && ch != '\'' {
            flush_non_ascii(&mut non_ascii, out);
            out.push(ch);
        } else {
            // Encode apostrophes too; this stays valid Part 21 and avoids
            // depending on every downstream reader handling doubled-apostrophe escaping correctly.
            non_ascii.push(ch);
        }
    }
    flush_non_ascii(&mut non_ascii, out);
    out.push('\'');
}

pub(super) fn format_real(v: f64) -> String {
    let mut s = v.to_string();
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push('.');
    }
    s
}

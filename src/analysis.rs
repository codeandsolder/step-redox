use crate::{
    complete_ir, formed_sheet, periodic_chains, solid_extrusions, solid_revolutions, step_io,
};
use anyhow::Result;
use ruststep::ast::Exchange;

/// Parsed STEP analysis context.
///
/// Construct this once when multiple read-only detectors are needed for the
/// same input. The convenience `*_bytes` APIs remain available, but each of
/// those necessarily owns its own parse.
pub struct AnalysisSession<'a> {
    input: &'a [u8],
    exchange: Exchange,
}

impl<'a> AnalysisSession<'a> {
    /// Decode and parse one supported STEP exchange.
    ///
    /// # Errors
    /// Returns an error when the input cannot be decoded or parsed, or when it
    /// contains optional exchange sections that step-redox does not support.
    pub fn from_bytes(input: &'a [u8]) -> Result<Self> {
        let exchange = step_io::ParsedExchange::parse(input)?.exchange;
        Ok(Self { input, exchange })
    }

    #[must_use]
    pub fn formed_sheet_evidence(&self) -> Vec<formed_sheet::FormedSheetEvidence> {
        self.exchange
            .data
            .iter()
            .flat_map(|section| formed_sheet::detect_formed_sheet_evidence(&section.entities))
            .collect()
    }

    #[must_use]
    pub fn solid_surface_signatures(&self) -> Vec<solid_revolutions::SolidSurfaceSignature> {
        self.exchange
            .data
            .iter()
            .flat_map(|section| {
                solid_revolutions::detect_solid_surface_signatures(&section.entities)
            })
            .collect()
    }

    #[must_use]
    pub fn solid_revolutions(&self) -> Vec<solid_revolutions::RecoveredSolidRevolution> {
        self.exchange
            .data
            .iter()
            .flat_map(|section| solid_revolutions::detect_solid_revolutions(&section.entities))
            .collect()
    }

    #[must_use]
    pub fn radial_slot_revolutions(&self) -> Vec<solid_revolutions::RecoveredRadialSlotRevolution> {
        self.exchange
            .data
            .iter()
            .flat_map(|section| {
                solid_revolutions::detect_radial_slot_revolutions(&section.entities)
            })
            .collect()
    }

    #[must_use]
    pub fn solid_extrusions(&self) -> Vec<solid_extrusions::RecoveredSolidExtrusion> {
        self.exchange
            .data
            .iter()
            .flat_map(|section| solid_extrusions::detect_solid_extrusions(&section.entities))
            .collect()
    }

    #[must_use]
    pub fn periodic_chains(&self) -> Vec<periodic_chains::PeriodicChainPattern> {
        self.exchange
            .data
            .iter()
            .flat_map(|section| periodic_chains::detect_periodic_chains(&section.entities))
            .collect()
    }

    /// Build the compact complete semantic IR without reparsing this exchange.
    ///
    /// # Errors
    /// Returns an error if detector output cannot be converted into exact CAD
    /// IR or solid ownership is ambiguous.
    pub fn complete_ir(&self) -> Result<complete_ir::CompleteIr> {
        complete_ir::recover_complete_ir_exchange(&self.exchange, self.input)
    }
}

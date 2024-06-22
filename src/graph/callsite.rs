extern crate backtrace;
use backtrace::{Backtrace, BacktraceFrame, BacktraceSymbol};
use std::fmt::{Display, Formatter, Result};

/// keeps track of where a call originated in the user's source for better error messages
#[derive(Debug, Clone)]
pub struct Callsite {
    /// value of file!() macro at callsite
    pub(crate) file: &'static str,
    /// value of line!() macro at callsite
    pub(crate) line: u32,
    /// unresolved call stack that got us here
    pub(crate) backtrace: Backtrace,
    /// how many levels up from here is the user code
    pub(crate) level: u32,
}

/// usage: `callsite!(1)` to generate a `Callsite` with level 1,
/// meaning that the function that called wherever this macro is used
/// would be referred to as the origin.
/// In deeper apis you can pass higher levels to still point at user code
macro_rules! callsite {
    ($( $arg:expr ),*) => {
        crate::graph::callsite::Callsite {
            file: file!(),
            line: line!(),
            backtrace: backtrace::Backtrace::new_unresolved(),
            level: $( $arg ),*
        }
    };
}
pub(crate) use callsite;

impl Display for Callsite {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let mut backtrace = self.backtrace.clone();
        backtrace.resolve();
        if let Some((file, line)) = backtrace
            .frames()
            .iter()
            .flat_map(BacktraceFrame::symbols)
            .skip_while(|s| {
                s.filename()
                    .map(|p| !p.ends_with(self.file))
                    .unwrap_or(true)
                    || s.lineno() != Some(self.line)
            })
            .nth(self.level as usize)
            .and_then(|symbol| {
                Some((
                    BacktraceSymbol::filename(symbol)?.to_str()?,
                    BacktraceSymbol::lineno(symbol)?,
                ))
            })
        {
            write!(f, "{}:{}", file, line)
        } else {
            // be silent if symbol resolve failed (we are probably in release mode)
            Ok(())
        }
    }
}

use super::*;

pub fn check_fp_type(dtype: xla::ElementType) -> Result<xla::ElementType> {
    match dtype {
        xla::ElementType::F16
        | xla::ElementType::Bf16
        | xla::ElementType::F32
        | xla::ElementType::F64
        | xla::ElementType::C64
        | xla::ElementType::C128 => Ok(dtype),
        _ => Err(ContextError::FPTypeError(dtype, callsite!(1))),
    }
}

pub fn check_int_type(dtype: xla::ElementType) -> Result<xla::ElementType> {
    match dtype {
        xla::ElementType::U8
        | xla::ElementType::S8
        | xla::ElementType::U16
        | xla::ElementType::S16
        | xla::ElementType::U32
        | xla::ElementType::S32
        | xla::ElementType::U64
        | xla::ElementType::S64 => Ok(dtype),
        _ => Err(ContextError::IntegralTypeError(dtype, callsite!(1))),
    }
}

pub fn check_real_type(dtype: xla::ElementType) -> Result<xla::ElementType> {
    match dtype {
        xla::ElementType::F16
        | xla::ElementType::Bf16
        | xla::ElementType::F32
        | xla::ElementType::F64 => Ok(dtype),
        _ => Err(ContextError::RealTypeError(dtype, callsite!(1))),
    }
}

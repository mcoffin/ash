pub trait NameExt {
    fn bits_to_flags(&self) -> String;
    fn strip_bits_suffix(&self) -> String;
    fn strip_vendor_suffix(&self, suffix: &str) -> Option<&str>;
}

impl NameExt for str {
    #[inline(always)]
    fn bits_to_flags(&self) -> String {
        self.replace("FlagBits", "Flags")
    }
    fn strip_bits_suffix(&self) -> String {
        self.replace("FlagBits", "")
    }
    #[inline(always)]
    fn strip_vendor_suffix(&self, suffix: &str) -> Option<&str> {
        self.strip_suffix(suffix)
            .and_then(|s| s.strip_suffix('_'))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnumType {
    Bitmask,
    Enum,
}

pub trait EnumsExt {
    fn ty(&self) -> Option<EnumType>;
}

impl EnumsExt for vk_parse::Enums {
    fn ty(&self) -> Option<EnumType> {
        match self.kind.as_deref() {
            Some("bitmask") => Some(EnumType::Bitmask),
            Some("enum") => Some(EnumType::Enum),
            _ => None,
        }
    }
}

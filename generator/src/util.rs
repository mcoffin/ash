use std::collections::HashSet;
use crate::{
    get_variant,
    config::ApiConfig,
};

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

#[derive(Debug)]
pub struct Requirements<'a> {
    pub types: HashSet<&'a str>,
    pub commands: HashSet<&'a str>,
    pub enums: HashSet<&'a str>,
}

impl<'a> Requirements<'a> {
    #[inline(always)]
    fn new() -> Self {
        Requirements {
            types: HashSet::new(),
            commands: HashSet::new(),
            enums: HashSet::new(),
        }
    }
}

impl<'a> From<&'a vk_parse::Registry> for Requirements<'a> {
    fn from(spec: &'a vk_parse::Registry) -> Self {
        unimplemented!()
    }
}

pub trait TypeMemberDefinitionExt {
    fn name(&self) -> Option<&str>;
    fn is_tagged_struct_type<C>(&self) -> bool
    where
        C: ApiConfig + ?Sized;
}

impl TypeMemberDefinitionExt for vk_parse::TypeMemberDefinition {
    fn name(&self) -> Option<&str> {
        self.markup.iter()
            .find_map(get_variant!(vk_parse::TypeMemberMarkup::Name))
            .map(AsRef::as_ref)
    }
    fn is_tagged_struct_type<C>(&self) -> bool
    where
        C: ApiConfig + ?Sized,
    {
        let Some(..) = self.markup.iter()
            .find_map(get_variant!(vk_parse::TypeMemberMarkup::Type))
            .filter(|&s| s == C::TAGGED_STRUCT.ty) else {
            return false;
        };
        self.markup.iter()
            .find_map(get_variant!(vk_parse::TypeMemberMarkup::Name))
            .filter(|&s| s == C::TAGGED_STRUCT.ty_name)
            .is_some()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TaggedStructInfo<'a, T> {
    pub type_field: &'a T,
    pub next_field: &'a T,
}

impl<'a, T: TaggedStructExt> TaggedStructInfo<'a, T> {
    #[inline(always)]
    pub fn get<C>(members: &'a [T]) -> Option<Self>
    where
        C: ApiConfig + ?Sized,
    {
        T::tagged_struct_info::<C>(members)
    }
}

pub trait TaggedStructExt: Sized {
    fn tagged_struct_info<C>(members: &[Self]) -> Option<TaggedStructInfo<'_, Self>>
    where
        C: ApiConfig + ?Sized;
}

impl<'a> TaggedStructExt for super::PreprocessedMember<'a> {
    fn tagged_struct_info<C>(members: &[Self]) -> Option<TaggedStructInfo<'_, Self>>
    where
        C: ApiConfig + ?Sized,
    {
        let vk_parse_members = || members.iter()
            .map(|v| (&v.vk_parse_type_member, v));

        let (_, ty) = vk_parse_members().find(|&(m, ..)| m.is_tagged_struct_type::<C>())?;
        let (_, next) = vk_parse_members().find(|&(m, ..)| {
            let v = m.markup.iter()
                .find_map(get_variant!(vk_parse::TypeMemberMarkup::Name));
            v.map(AsRef::as_ref) == Some(C::TAGGED_STRUCT.next_name)
        })?;
        Some(TaggedStructInfo {
            type_field: ty,
            next_field: next,
        })
    }
}

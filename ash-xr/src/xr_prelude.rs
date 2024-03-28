use crate::xr;

pub unsafe trait TaggedStructure {
    const STRUCTURE_TYPE: xr::StructureType;
}

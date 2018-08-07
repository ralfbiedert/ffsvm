use super::Simd;

crate fn simd_vector_to_flat_slice<SimdType>(
    slice: &[SimdType],
    length: usize,
) -> &[SimdType::Element]
where
    SimdType: Simd + Default + Clone,
{
    let ptr = slice.as_ptr() as *const SimdType::Element;

    // This "should be safe(tm)" since:
    //
    // 1) a slice of `N x f32x8` elements are transformed into a slice of
    // `attributes * f32` elements, where `attributes <=  N * 8`.
    //
    // 2) The lifetime of the returned value should automatically match the self borrow.
    //
    // Having said this, as soon as `std::simd` (or similar) provides a safe way of handling
    // that for us, these lines should be removed!
    unsafe { std::slice::from_raw_parts(ptr, length) }
}

crate fn simd_vector_to_flat_slice_mut<SimdType>(
    slice: &mut [SimdType],
    length: usize,
) -> &mut [SimdType::Element]
where
    SimdType: Simd + Default + Clone,
{
    let mut_ptr = slice.as_mut_ptr() as *mut SimdType::Element;

    // See comment above
    unsafe { std::slice::from_raw_parts_mut(mut_ptr, length) }
}

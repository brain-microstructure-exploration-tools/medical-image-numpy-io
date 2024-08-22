from typing import Union, Iterable, Any, Sequence, Collection, Hashable
import os
import re
from enum import StrEnum
import numpy as np
from pathlib import PurePath

# General filepath type
PathLike = Union[str, os.PathLike]

#: Type of datatypes: Adapted from https://github.com/numpy/numpy/blob/v1.21.4/numpy/typing/_dtype_like.py#L121
DtypeLike = Union[np.dtype, type, str, None]

# The KeyCollection type is used to for defining variables
# that store a subset of keys to select items from a dictionary.
# The container of keys must contain hashable elements.
# NOTE:  `Hashable` is not a collection, but is provided as a
#        convenience to end-users.  All supplied values will be
#        internally converted to a tuple of `Hashable`'s before
#        use
KeysCollection = Union[Collection[Hashable], Hashable]

# Pattern to pick out certain numpy dtypes; taken from torch.utils.data._utils.collate
np_str_obj_array_pattern = re.compile(r"[SaUO]")

def is_no_channel(val) -> bool:
    """Returns whether `val` indicates "no_channel", for MetaKeys.ORIGINAL_CHANNEL_DIM."""
    if isinstance(val, str):
        return val == "no_channel"
    if np.isscalar(val):
        return bool(np.isnan(val))
    return val is None

def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    try:
        if hasattr(obj, "ndim") and obj.ndim == 0:
            return False  # a 0-d tensor is not iterable
    except Exception:
        return False
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))

def ensure_tuple(vals: Any, wrap_array: bool = False) -> tuple:
    """
    Returns a tuple of `vals`.

    Args:
        vals: input data to convert to a tuple.
        wrap_array: if `True`, treat the input numerical array (ndarray/tensor) as one item of the tuple.
            if `False`, try to convert the array with `tuple(vals)`, default to `False`.

    """
    if wrap_array and isinstance(vals, np.ndarray):
        return (vals,)
    return tuple(vals) if issequenceiterable(vals) else (vals,)

def orientation_ras_lps(affine: np.ndarray) -> np.ndarray:
    """
    Convert the ``affine`` between the `RAS` and `LPS` orientation
    by flipping the first two spatial dimensions.

    Args:
        affine: a 2D affine matrix.
    """
    sr = max(affine.shape[0] - 1, 1)  # spatial rank is at least 1
    flip_d = [[-1, 1], [-1, -1, 1], [-1, -1, 1, 1]]
    flip_diag = flip_d[min(sr - 1, 2)] + [1] * (sr - 3)
    return np.diag(flip_diag).astype(affine.dtype) @ affine  # type: ignore

def get_dtype_bound_value(dtype: DtypeLike) -> tuple[float, float]:
    """
    Get dtype bound value
    Args:
        dtype: dtype to get bound value
    Returns:
        (bound_min_value, bound_max_value)
    """
    if dtype.is_floating_point:
        return (np.finfo(dtype).min, np.finfo(dtype).max)  # type: ignore
    else:
        return (np.iinfo(dtype).min, np.iinfo(dtype).max)

def safe_dtype_range(data: Any, dtype: DtypeLike = None) -> Any:
    """
    Utility to safely convert the input data to target dtype.

    Args:
        data: input data can be numpy array, list, dictionary, int, float, bool, str, etc.
            will convert to target dtype and keep the original type.
            for dictionary, list or tuple, convert every item.
        dtype: target data type to convert.
    """

    def _safe_dtype_range(data, dtype):
        output_dtype = dtype if dtype is not None else data.dtype
        dtype_bound_value = get_dtype_bound_value(output_dtype)
        if data.ndim == 0:
            data_bound = (data, data)
        else:
            data_bound = (np.min(data), np.max(data))
        if (data_bound[1] > dtype_bound_value[1]) or (data_bound[0] < dtype_bound_value[0]):
            if isinstance(data, np.ndarray):
                return np.clip(data, dtype_bound_value[0], dtype_bound_value[1])
        else:
            return data

    if isinstance(data, np.ndarray):
        return np.asarray(_safe_dtype_range(data, dtype))
    elif isinstance(data, (float, int, bool)) and dtype is None:
        return data
    elif isinstance(data, (float, int, bool)) and dtype is not None:
        output_dtype = dtype
        dtype_bound_value = get_dtype_bound_value(output_dtype)
        data = dtype_bound_value[1] if data > dtype_bound_value[1] else data
        data = dtype_bound_value[0] if data < dtype_bound_value[0] else data
        return data
    elif isinstance(data, list):
        return [safe_dtype_range(i, dtype=dtype) for i in data]
    elif isinstance(data, tuple):
        return tuple(safe_dtype_range(i, dtype=dtype) for i in data)
    elif isinstance(data, dict):
        return {k: safe_dtype_range(v, dtype=dtype) for k, v in data.items()}
    return data

def convert_to_numpy(data: Any, dtype: DtypeLike = None, wrap_sequence: bool = False, safe: bool = False) -> Any:
    """
    Utility to convert the input data to a numpy array. If passing a dictionary, list or tuple,
    recursively check every item and convert it to numpy array.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to numpy arrays, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a numpy array if applicable.
        dtype: target data type when converting to numpy array.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then `[256, -12]` -> `[array(255), array(0)]`.
    """
    if safe:
        data = safe_dtype_range(data, dtype)
    if isinstance(data, (np.ndarray, float, int, bool)):
        # Convert into a contiguous array first if the current dtype's size is smaller than the target dtype's size.
        # This help improve the performance because (convert to contiguous array) -> (convert dtype) is faster
        # than (convert dtype) -> (convert to contiguous array) when src dtype (e.g., uint8) is smaller than
        # target dtype(e.g., float32) and we are going to convert it to contiguous array anyway later in this
        # method.
        if isinstance(data, np.ndarray) and data.ndim > 0 and data.dtype.itemsize < np.dtype(dtype).itemsize:
            data = np.ascontiguousarray(data)
        data = np.asarray(data, dtype=dtype)
    elif isinstance(data, list):
        list_ret = [convert_to_numpy(i, dtype=dtype) for i in data]
        return np.asarray(list_ret) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_numpy(i, dtype=dtype) for i in data)
        return np.asarray(tuple_ret) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_numpy(v, dtype=dtype) for k, v in data.items()}

    if isinstance(data, np.ndarray) and data.ndim > 0:
        data = np.ascontiguousarray(data)

    return data

def convert_data_type(
    data: Any,
    output_type: type[np.ndarray] | None = None,
    dtype: DtypeLike = None,
    wrap_sequence: bool = False,
    safe: bool = False,
) -> tuple[np.ndarray, type]:
    """
    See monai.utils.type_conversion.convert_data_type.
    This is that function but stripped of torch and cupy.
    """
    orig_type: type
    if isinstance(data, np.ndarray):
        orig_type = np.ndarray
    else:
        orig_type = type(data)

    output_type = output_type or orig_type
    dtype_ = dtype # Was get_equivalent_dtype(dtype, output_type), but after stripping out torch it reduces to just dtype.

    data_: np.ndarray
    if issubclass(output_type, np.ndarray):
        data_ = convert_to_numpy(data, dtype=dtype_, wrap_sequence=wrap_sequence, safe=safe)
        return data_, orig_type
    raise ValueError(f"Unsupported output type: {output_type}")

def convert_to_dst_type(
    src: Any,
    dst: np.ndarray,
    dtype: DtypeLike  | None = None,
    wrap_sequence: bool = False,
    safe: bool = False,
) -> tuple[np.ndarray, type]:
    """
    Convert source data to the same data type as the destination data.

    Args:
        src: source data to convert type.
        dst: destination data that convert to the same data type as it.
        dtype: an optional argument if the target `dtype` is different from the original `dst`'s data type.
        wrap_sequence: if `False`, then lists will recursively call this function. E.g., `[1, 2]` -> `[array(1), array(2)]`.
            If `True`, then `[1, 2]` -> `array([1, 2])`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then `[256, -12]` -> `[array(255), array(0)]`.

    See Also:
        :func:`convert_data_type`
    """

    if dtype is None:
        dtype = getattr(dst, "dtype", None)  # sequence has no dtype

    output_type: Any
    if isinstance(dst, np.ndarray):
        output_type = np.ndarray
    else:
        output_type = type(dst)
    output: np.ndarray
    return convert_data_type(
        data=src, output_type=output_type, dtype=dtype, wrap_sequence=wrap_sequence, safe=safe
    )

def affine_to_spacing(affine: np.ndarray, r: int = 3, dtype=float, suppress_zeros: bool = True) -> np.ndarray:
    """
    Computing the current spacing from the affine matrix.

    Args:
        affine: a d x d affine matrix.
        r: indexing based on the spatial rank, spacing is computed from `affine[:r, :r]`.
        dtype: data type of the output.
        suppress_zeros: whether to suppress the zeros with ones.

    Returns:
        an `r` dimensional vector of spacing.
    """
    if len(affine.shape) != 2 or affine.shape[0] != affine.shape[1]:
        raise ValueError(f"affine must be a square matrix, got {affine.shape}.")
    _affine, *_ = convert_to_dst_type(affine[:r, :r], dst=affine, dtype=dtype)
    spacing = np.sqrt(np.sum(_affine * _affine, axis=0))
    if suppress_zeros:
        spacing[spacing == 0] = 1.0
    spacing_, *_ = convert_to_dst_type(spacing, dst=affine, dtype=dtype)
    return spacing_

def is_supported_format(filename: Sequence[PathLike] | PathLike, suffixes: Sequence[str]) -> bool:
    """
    Verify whether the specified file or files format match supported suffixes.
    If supported suffixes is None, skip the verification and return True.

    Args:
        filename: file name or a list of file names to read.
            if a list of files, verify all the suffixes.
        suffixes: all the supported image suffixes of current reader, must be a list of lower case suffixes.

    """
    filenames: Sequence[PathLike] = ensure_tuple(filename)
    for name in filenames:
        full_suffix = "".join(map(str.lower, PurePath(name).suffixes))
        if all(f".{s.lower()}" not in full_suffix for s in suffixes):
            return False

    return True

def rectify_header_sform_qform(img_nii):
    """
    Look at the sform and qform of the nifti object and correct it if any
    incompatibilities with pixel dimensions

    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/io/misc_io.py

    Args:
        img_nii: nifti image object
    """
    d = img_nii.header["dim"][0]
    pixdim = np.asarray(img_nii.header.get_zooms())[:d]
    sform, qform = img_nii.get_sform(), img_nii.get_qform()
    norm_sform = affine_to_spacing(sform, r=d)
    norm_qform = affine_to_spacing(qform, r=d)
    sform_mismatch = not np.allclose(norm_sform, pixdim)
    qform_mismatch = not np.allclose(norm_qform, pixdim)

    if img_nii.header["sform_code"] != 0:
        if not sform_mismatch:
            return img_nii
        if not qform_mismatch:
            img_nii.set_sform(img_nii.get_qform())
            return img_nii
    if img_nii.header["qform_code"] != 0:
        if not qform_mismatch:
            return img_nii
        if not sform_mismatch:
            img_nii.set_qform(img_nii.get_sform())
            return img_nii

    norm = affine_to_spacing(img_nii.affine, r=d)

    img_nii.header.set_zooms(norm)
    return img_nii

def correct_nifti_header_if_necessary(img_nii):
    """
    Check nifti object header's format, update the header if needed.
    In the updated image pixdim matches the affine.

    Args:
        img_nii: nifti image object
    """
    if img_nii.header.get("dim") is None:
        return img_nii  # not nifti?
    dim = img_nii.header["dim"][0]
    if dim >= 5:
        return img_nii  # do nothing for high-dimensional array
    # check that affine matches zooms
    pixdim = np.asarray(img_nii.header.get_zooms())[:dim]
    norm_affine = affine_to_spacing(img_nii.affine, r=dim)
    if np.allclose(pixdim, norm_affine):
        return img_nii
    if hasattr(img_nii, "get_sform"):
        return rectify_header_sform_qform(img_nii)
    return img_nii

class SpaceKeys(StrEnum):
    """
    The coordinate system keys, for example, Nifti1 uses Right-Anterior-Superior or "RAS",
    DICOM (0020,0032) uses Left-Posterior-Superior or "LPS". This type does not distinguish spatial 1/2/3D.
    """

    RAS = "RAS"
    LPS = "LPS"

class MetaKeys(StrEnum):
    """
    Typical keys for MetaObj.meta
    """

    AFFINE = "affine"  # MetaTensor.affine
    ORIGINAL_AFFINE = "original_affine"  # the affine after image loading before any data processing
    SPATIAL_SHAPE = "spatial_shape"  # optional key for the length in each spatial dimension
    SPACE = "space"  # possible values of space type are defined in `SpaceKeys`
    ORIGINAL_CHANNEL_DIM = "original_channel_dim"  # an integer or float("nan")

class TraceKeys(StrEnum):
    """Extra metadata keys used for traceable transforms."""

    CLASS_NAME: str = "class"
    ID: str = "id"
    ORIG_SIZE: str = "orig_size"
    EXTRA_INFO: str = "extra_info"
    DO_TRANSFORM: str = "do_transforms"
    KEY_SUFFIX: str = "_transforms"
    NONE: str = "none"
    TRACING: str = "tracing"
    STATUSES: str = "statuses"
    LAZY: str = "lazy"
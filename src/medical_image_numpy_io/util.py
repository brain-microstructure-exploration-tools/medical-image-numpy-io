from typing import Union, Iterable, Any, Sequence, Collection, Hashable
import os
import re
from enum import StrEnum, EnumMeta, Enum
from collections.abc import Mapping
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

class OptionalImportError(ImportError):
    """
    Could not import APIs from an optional dependency.
    """

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

class GridSampleMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

    interpolation mode of `torch.nn.functional.grid_sample`

    Note:
        (documentation from `torch.nn.functional.grid_sample`)
        `mode='bicubic'` supports only 4-D input.
        When `mode='bilinear'` and the input is 5-D, the interpolation mode used internally will actually be trilinear.
        However, when the input is 4-D, the interpolation mode will legitimately be bilinear.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"

class InterpolateMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    AREA = "area"    

class GridSamplePadMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    """

    ZEROS = "zeros"
    BORDER = "border"
    REFLECTION = "reflection"

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

def ensure_tuple_rep(tup: Any, dim: int) -> tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")

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

def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculates the Damerau–Levenshtein distance between two strings for spelling correction.
    https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance
    """
    if s1 == s2:
        return 0
    string_1_length = len(s1)
    string_2_length = len(s2)
    if not s1:
        return string_2_length
    if not s2:
        return string_1_length
    d = {(i, -1): i + 1 for i in range(-1, string_1_length + 1)}
    for j in range(-1, string_2_length + 1):
        d[(-1, j)] = j + 1

    for i, s1i in enumerate(s1):
        for j, s2j in enumerate(s2):
            cost = 0 if s1i == s2j else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1, d[(i, j - 1)] + 1, d[(i - 1, j - 1)] + cost  # deletion  # insertion  # substitution
            )
            if i and j and s1i == s2[j - 1] and s1[i - 1] == s2j:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[string_1_length - 1, string_2_length - 1]

def look_up_option(
    opt_str: Hashable,
    supported: Collection | EnumMeta,
    default: Any = "no_default",
    print_all_options: bool = True,
) -> Any:
    """
    Look up the option in the supported collection and return the matched item.
    Raise a value error possibly with a guess of the closest match.

    Args:
        opt_str: The option string or Enum to look up.
        supported: The collection of supported options, it can be list, tuple, set, dict, or Enum.
        default: If it is given, this method will return `default` when `opt_str` is not found,
            instead of raising a `ValueError`. Otherwise, it defaults to `"no_default"`,
            so that the method may raise a `ValueError`.
        print_all_options: whether to print all available options when `opt_str` is not found. Defaults to True

    Examples:

    .. code-block:: python

        from enum import Enum
        from monai.utils import look_up_option
        class Color(Enum):
            RED = "red"
            BLUE = "blue"
        look_up_option("red", Color)  # <Color.RED: 'red'>
        look_up_option(Color.RED, Color)  # <Color.RED: 'red'>
        look_up_option("read", Color)
        # ValueError: By 'read', did you mean 'red'?
        # 'read' is not a valid option.
        # Available options are {'blue', 'red'}.
        look_up_option("red", {"red", "blue"})  # "red"

    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/utilities/util_common.py#L249
    """
    if not isinstance(opt_str, Hashable):
        raise ValueError(f"Unrecognized option type: {type(opt_str)}:{opt_str}.")
    if isinstance(opt_str, str):
        opt_str = opt_str.strip()
    if isinstance(supported, EnumMeta):
        if isinstance(opt_str, str) and opt_str in {item.value for item in supported}:  # type: ignore
            # such as: "example" in MyEnum
            return supported(opt_str)
        if isinstance(opt_str, Enum) and opt_str in supported:
            # such as: MyEnum.EXAMPLE in MyEnum
            return opt_str
    elif isinstance(supported, Mapping) and opt_str in supported:
        # such as: MyDict[key]
        return supported[opt_str]
    elif isinstance(supported, Collection) and opt_str in supported:
        return opt_str

    if default != "no_default":
        return default

    # find a close match
    set_to_check: set
    if isinstance(supported, EnumMeta):
        set_to_check = {item.value for item in supported}  # type: ignore
    else:
        set_to_check = set(supported) if supported is not None else set()
    if not set_to_check:
        raise ValueError(f"No options available: {supported}.")
    edit_dists = {}
    opt_str = f"{opt_str}"
    for key in set_to_check:
        edit_dist = damerau_levenshtein_distance(f"{key}", opt_str)
        if edit_dist <= 3:
            edit_dists[key] = edit_dist

    supported_msg = f"Available options are {set_to_check}.\n" if print_all_options else ""
    if edit_dists:
        guess_at_spelling = min(edit_dists, key=edit_dists.get)  # type: ignore
        raise ValueError(
            f"By '{opt_str}', did you mean '{guess_at_spelling}'?\n"
            + f"'{opt_str}' is not a valid value.\n"
            + supported_msg
        )
    raise ValueError(f"Unsupported option '{opt_str}', " + supported_msg)

def to_affine_nd(r: np.ndarray | int, affine: np.ndarray, dtype=np.float64) -> np.ndarray:
    """
    Using elements from affine, to create a new affine matrix by
    assigning the rotation/zoom/scaling matrix and the translation vector.

    When ``r`` is an integer, output is an (r+1)x(r+1) matrix,
    where the top left kxk elements are copied from ``affine``,
    the last column of the output affine is copied from ``affine``'s last column.
    `k` is determined by `min(r, len(affine) - 1)`.

    When ``r`` is an affine matrix, the output has the same shape as ``r``,
    and the top left kxk elements are copied from ``affine``,
    the last column of the output affine is copied from ``affine``'s last column.
    `k` is determined by `min(len(r) - 1, len(affine) - 1)`.

    Args:
        r (int or matrix): number of spatial dimensions or an output affine to be filled.
        affine (matrix): 2D affine matrix
        dtype: data type of the output array.

    Raises:
        ValueError: When ``affine`` dimensions is not 2.
        ValueError: When ``r`` is nonpositive.

    Returns:
        an (r+1) x (r+1) matrix (tensor or ndarray depends on the input ``affine`` data type)

    """
    affine_np = convert_data_type(affine, output_type=np.ndarray, dtype=dtype, wrap_sequence=True)[0]
    affine_np = affine_np.copy()
    if affine_np.ndim != 2:
        raise ValueError(f"affine must have 2 dimensions, got {affine_np.ndim}.")
    new_affine = np.array(r, dtype=dtype, copy=True)
    if new_affine.ndim == 0:
        sr: int = int(new_affine.astype(np.uint))
        if not np.isfinite(sr) or sr < 0:
            raise ValueError(f"r must be positive, got {sr}.")
        new_affine = np.eye(sr + 1, dtype=dtype)
    d = max(min(len(new_affine) - 1, len(affine_np) - 1), 1)
    new_affine[:d, :d] = affine_np[:d, :d]
    if d > 1:
        new_affine[:d, -1] = affine_np[:d, -1]
    output, *_ = convert_to_dst_type(new_affine, affine, dtype=dtype)
    return output

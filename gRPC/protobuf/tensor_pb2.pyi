import tensor_shape_pb2 as _tensor_shape_pb2
import types_pb2 as _types_pb2
import resource_handle_pb2 as _resource_handle_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorProto(_message.Message):
    __slots__ = ("dtype", "tensor_shape", "version_number", "tensor_content", "half_val", "float_val", "double_val", "int_val", "string_val", "scomplex_val", "int64_val", "bool_val", "dcomplex_val", "resource_handle_val", "variant_val", "uint32_val", "uint64_val")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SHAPE_FIELD_NUMBER: _ClassVar[int]
    VERSION_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TENSOR_CONTENT_FIELD_NUMBER: _ClassVar[int]
    HALF_VAL_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VAL_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_VAL_FIELD_NUMBER: _ClassVar[int]
    INT_VAL_FIELD_NUMBER: _ClassVar[int]
    STRING_VAL_FIELD_NUMBER: _ClassVar[int]
    SCOMPLEX_VAL_FIELD_NUMBER: _ClassVar[int]
    INT64_VAL_FIELD_NUMBER: _ClassVar[int]
    BOOL_VAL_FIELD_NUMBER: _ClassVar[int]
    DCOMPLEX_VAL_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_HANDLE_VAL_FIELD_NUMBER: _ClassVar[int]
    VARIANT_VAL_FIELD_NUMBER: _ClassVar[int]
    UINT32_VAL_FIELD_NUMBER: _ClassVar[int]
    UINT64_VAL_FIELD_NUMBER: _ClassVar[int]
    dtype: _types_pb2.DataType
    tensor_shape: _tensor_shape_pb2.TensorShapeProto
    version_number: int
    tensor_content: bytes
    half_val: _containers.RepeatedScalarFieldContainer[int]
    float_val: _containers.RepeatedScalarFieldContainer[float]
    double_val: _containers.RepeatedScalarFieldContainer[float]
    int_val: _containers.RepeatedScalarFieldContainer[int]
    string_val: _containers.RepeatedScalarFieldContainer[bytes]
    scomplex_val: _containers.RepeatedScalarFieldContainer[float]
    int64_val: _containers.RepeatedScalarFieldContainer[int]
    bool_val: _containers.RepeatedScalarFieldContainer[bool]
    dcomplex_val: _containers.RepeatedScalarFieldContainer[float]
    resource_handle_val: _containers.RepeatedCompositeFieldContainer[_resource_handle_pb2.ResourceHandleProto]
    variant_val: _containers.RepeatedCompositeFieldContainer[VariantTensorDataProto]
    uint32_val: _containers.RepeatedScalarFieldContainer[int]
    uint64_val: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, dtype: _Optional[_Union[_types_pb2.DataType, str]] = ..., tensor_shape: _Optional[_Union[_tensor_shape_pb2.TensorShapeProto, _Mapping]] = ..., version_number: _Optional[int] = ..., tensor_content: _Optional[bytes] = ..., half_val: _Optional[_Iterable[int]] = ..., float_val: _Optional[_Iterable[float]] = ..., double_val: _Optional[_Iterable[float]] = ..., int_val: _Optional[_Iterable[int]] = ..., string_val: _Optional[_Iterable[bytes]] = ..., scomplex_val: _Optional[_Iterable[float]] = ..., int64_val: _Optional[_Iterable[int]] = ..., bool_val: _Optional[_Iterable[bool]] = ..., dcomplex_val: _Optional[_Iterable[float]] = ..., resource_handle_val: _Optional[_Iterable[_Union[_resource_handle_pb2.ResourceHandleProto, _Mapping]]] = ..., variant_val: _Optional[_Iterable[_Union[VariantTensorDataProto, _Mapping]]] = ..., uint32_val: _Optional[_Iterable[int]] = ..., uint64_val: _Optional[_Iterable[int]] = ...) -> None: ...

class VariantTensorDataProto(_message.Message):
    __slots__ = ("type_name", "metadata", "tensors")
    TYPE_NAME_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    type_name: str
    metadata: bytes
    tensors: _containers.RepeatedCompositeFieldContainer[TensorProto]
    def __init__(self, type_name: _Optional[str] = ..., metadata: _Optional[bytes] = ..., tensors: _Optional[_Iterable[_Union[TensorProto, _Mapping]]] = ...) -> None: ...

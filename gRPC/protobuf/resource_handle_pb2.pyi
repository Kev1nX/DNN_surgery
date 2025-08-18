import tensor_shape_pb2 as _tensor_shape_pb2
import types_pb2 as _types_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ResourceHandleProto(_message.Message):
    __slots__ = ("device", "container", "name", "hash_code", "maybe_type_name", "dtypes_and_shapes")
    class DtypeAndShape(_message.Message):
        __slots__ = ("dtype", "shape")
        DTYPE_FIELD_NUMBER: _ClassVar[int]
        SHAPE_FIELD_NUMBER: _ClassVar[int]
        dtype: _types_pb2.DataType
        shape: _tensor_shape_pb2.TensorShapeProto
        def __init__(self, dtype: _Optional[_Union[_types_pb2.DataType, str]] = ..., shape: _Optional[_Union[_tensor_shape_pb2.TensorShapeProto, _Mapping]] = ...) -> None: ...
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    HASH_CODE_FIELD_NUMBER: _ClassVar[int]
    MAYBE_TYPE_NAME_FIELD_NUMBER: _ClassVar[int]
    DTYPES_AND_SHAPES_FIELD_NUMBER: _ClassVar[int]
    device: str
    container: str
    name: str
    hash_code: int
    maybe_type_name: str
    dtypes_and_shapes: _containers.RepeatedCompositeFieldContainer[ResourceHandleProto.DtypeAndShape]
    def __init__(self, device: _Optional[str] = ..., container: _Optional[str] = ..., name: _Optional[str] = ..., hash_code: _Optional[int] = ..., maybe_type_name: _Optional[str] = ..., dtypes_and_shapes: _Optional[_Iterable[_Union[ResourceHandleProto.DtypeAndShape, _Mapping]]] = ...) -> None: ...

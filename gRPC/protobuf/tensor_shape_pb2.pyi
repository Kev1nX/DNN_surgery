from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorShapeProto(_message.Message):
    __slots__ = ("dim", "unknown_rank")
    class Dim(_message.Message):
        __slots__ = ("size", "name")
        SIZE_FIELD_NUMBER: _ClassVar[int]
        NAME_FIELD_NUMBER: _ClassVar[int]
        size: int
        name: str
        def __init__(self, size: _Optional[int] = ..., name: _Optional[str] = ...) -> None: ...
    DIM_FIELD_NUMBER: _ClassVar[int]
    UNKNOWN_RANK_FIELD_NUMBER: _ClassVar[int]
    dim: _containers.RepeatedCompositeFieldContainer[TensorShapeProto.Dim]
    unknown_rank: bool
    def __init__(self, dim: _Optional[_Iterable[_Union[TensorShapeProto.Dim, _Mapping]]] = ..., unknown_rank: bool = ...) -> None: ...

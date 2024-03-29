# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow_manager/proto/hyperparams/hyperparams.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow_manager/proto/hyperparams/hyperparams.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n6tensorflow_manager/proto/hyperparams/hyperparams.proto\"s\n\x0bHyperParams\x12\'\n\x04loss\x18\x01 \x01(\x0e\x32\x19.HyperParams.LossFunction\x12\x12\n\nfromLogits\x18\x02 \x01(\x08\"\'\n\x0cLossFunction\x12\x17\n\x13SparseCategoricalCE\x10\x00\x62\x06proto3'
)



_HYPERPARAMS_LOSSFUNCTION = _descriptor.EnumDescriptor(
  name='LossFunction',
  full_name='HyperParams.LossFunction',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SparseCategoricalCE', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=134,
  serialized_end=173,
)
_sym_db.RegisterEnumDescriptor(_HYPERPARAMS_LOSSFUNCTION)


_HYPERPARAMS = _descriptor.Descriptor(
  name='HyperParams',
  full_name='HyperParams',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='loss', full_name='HyperParams.loss', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='fromLogits', full_name='HyperParams.fromLogits', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _HYPERPARAMS_LOSSFUNCTION,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=58,
  serialized_end=173,
)

_HYPERPARAMS.fields_by_name['loss'].enum_type = _HYPERPARAMS_LOSSFUNCTION
_HYPERPARAMS_LOSSFUNCTION.containing_type = _HYPERPARAMS
DESCRIPTOR.message_types_by_name['HyperParams'] = _HYPERPARAMS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

HyperParams = _reflection.GeneratedProtocolMessageType('HyperParams', (_message.Message,), {
  'DESCRIPTOR' : _HYPERPARAMS,
  '__module__' : 'tensorflow_manager.proto.hyperparams.hyperparams_pb2'
  # @@protoc_insertion_point(class_scope:HyperParams)
  })
_sym_db.RegisterMessage(HyperParams)


# @@protoc_insertion_point(module_scope)
